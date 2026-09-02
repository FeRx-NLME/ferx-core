//! The [`ModelEdit`] appliers.
//!
//! Every variant follows the same shape: **validate everything first, mutate
//! second**. A rejected edit must leave the text exactly as it found it, or a
//! search loop that recovers from a bad proposal would carry a half-written
//! model into the next candidate.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use super::spec::{finite, num, ErrorSpecText, IivForm, ModelEdit, Relation, StructuralSpec};
use super::ModelText;
use crate::types::{FitResult, PkModel};

// ── Line grammars ───────────────────────────────────────────────────────────
//
// Deliberately the *editing* subset of `parse_parameters`' regexes: enough to
// find a declaration and splice its initial value, never enough to be a second
// parser. Anything these do not match is left alone and reported, not guessed.

/// `theta NAME(init …` — group 1 is everything up to the init, group 2 the
/// name, group 3 the init, group 4 the rest (bounds, `FIX`, `)`).
static THETA_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(theta\s+(\w+)\s*\(\s*)([0-9eE.+-]+)(.*)$").unwrap());
/// `omega NAME ~ value …`, same group layout.
static OMEGA_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(omega\s+(\w+)\s*~\s*)([0-9eE.+-]+)(.*)$").unwrap());
/// `sigma NAME ~ value …`, same group layout.
static SIGMA_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(sigma\s+(\w+)\s*~\s*)([0-9eE.+-]+)(.*)$").unwrap());
/// `block_omega (A, B) = [ … ]` — group 2 the names, group 3 the triangle.
static BLOCK_OMEGA_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(block_omega\s*\(([^)]*)\)\s*=\s*)\[([^\]]*)\](.*)$").unwrap()
});
/// `block_sigma (A, B) = [ … ]` — detection only; this module never edits one.
static BLOCK_SIGMA_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^block_sigma\s*\(").unwrap());
/// `NAME = <expr>` — an `[individual_parameters]` assignment. The `[^=]`
/// lookahead keeps `==` out.
static ASSIGN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^([A-Za-z_]\w*)\s*=\s*([^=].*)$").unwrap());
/// `pk NAME(...)` in `[structural_model]`.
static PK_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^pk\s+(\w+)\s*\(").unwrap());
/// A `FIX` keyword anywhere on the line, as its own token.
static FIX_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\bFIX\b").unwrap());
/// The `(sd)` scale annotation — `(variance)`/`(var)` are the default and need
/// no detection.
static SD_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\(\s*sd\s*\)").unwrap());

pub(crate) fn apply(text: &mut ModelText, edit: ModelEdit<'_>) -> Result<(), String> {
    match edit {
        ModelEdit::SetStructural(spec) => set_structural(text, &spec),
        ModelEdit::AddCovariateRelation(rel) => add_covariate_relation(text, &rel),
        ModelEdit::DropCovariateRelation { param, cov } => {
            drop_covariate_relation(text, &param, &cov)
        }
        ModelEdit::AddIiv { param, form } => add_iiv(text, &param, &form),
        ModelEdit::DropIiv { param } => drop_iiv(text, &param),
        ModelEdit::SetOmegaBlock(names) => set_omega_block(text, &names),
        ModelEdit::SetErrorModel(spec) => set_error_model(text, &spec),
        ModelEdit::SeedInits(fit) => seed_inits(text, fit),
        ModelEdit::SetFitOption { key, value } => set_fit_option(text, &key, &value),
    }
}

// ── [structural_model] ──────────────────────────────────────────────────────

fn set_structural(text: &mut ModelText, spec: &StructuralSpec) -> Result<(), String> {
    if PkModel::from_name(&spec.template).is_none() {
        return Err(format!(
            "ferx-core::edit: `{}` is not a known `pk` template",
            spec.template
        ));
    }
    if spec.bindings.is_empty() {
        return Err(format!(
            "ferx-core::edit: `pk {}()` binds no parameters",
            spec.template
        ));
    }
    for p in &spec.new_parameters {
        finite(p.init, &format!("`{}` init", p.theta))?;
        finite(p.lower, &format!("`{}` lower bound", p.theta))?;
        finite(p.upper, &format!("`{}` upper bound", p.theta))?;
        if let Some((eta, v)) = &p.iiv {
            finite(*v, &format!("`omega {eta}` variance"))?;
        }
    }

    let Some(pk_line) = find_line(text, "structural_model", |c| PK_RE.is_match(c)) else {
        return Err(
            "ferx-core::edit: SetStructural needs a `pk NAME(...)` line in [structural_model]. \
             An `ode(...)` or algebraic model has no template to swap — edit its [odes] block \
             directly."
                .to_string(),
        );
    };

    // Every binding target must either exist already or arrive with a default.
    let existing = individual_parameter_names(text);
    let mut to_create = Vec::new();
    for (role, var) in &spec.bindings {
        if existing.contains_key(var) {
            continue;
        }
        match spec.new_parameters.iter().find(|p| &p.name == var) {
            Some(p) => to_create.push(p),
            None => {
                return Err(format!(
                    "ferx-core::edit: `pk {}({role}={var})` names `{var}`, which the model does \
                     not declare in [individual_parameters]. Supply its init and bounds in \
                     `StructuralSpec::new_parameters` — a widening search has to choose them, \
                     and choosing them here keeps the choice visible per candidate.",
                    spec.template
                ))
            }
        }
    }

    // ── Mutate ──
    let pairs: Vec<String> = spec
        .bindings
        .iter()
        .map(|(role, var)| format!("{role}={var}"))
        .collect();
    text.set_code(
        pk_line,
        &format!("pk {}({})", spec.template, pairs.join(", ")),
    );

    for p in to_create {
        let rhs = match &p.iiv {
            Some((eta, _)) => format!("{} * exp({eta})", p.theta),
            None => p.theta.clone(),
        };
        text.append_to_block("individual_parameters", &format!("{} = {rhs}", p.name));
        text.append_to_block(
            "parameters",
            &format!(
                "theta {}({}, {}, {})",
                p.theta,
                num(p.init),
                num(p.lower),
                num(p.upper)
            ),
        );
        if let Some((eta, v)) = &p.iiv {
            text.append_to_block("parameters", &format!("omega {eta} ~ {}", num(*v)));
        }
    }

    prune_unreferenced(text)
}

// ── [covariate_model] ───────────────────────────────────────────────────────

fn add_covariate_relation(text: &mut ModelText, rel: &Relation) -> Result<(), String> {
    for t in &rel.thetas {
        finite(t.init, &format!("`{}` init", t.name))?;
        finite(t.lower, &format!("`{}` lower bound", t.name))?;
        finite(t.upper, &format!("`{}` upper bound", t.name))?;
    }
    if let Some(v) = rel.fix {
        finite(v, "`fix`")?;
    }
    if find_relation_line(text, &rel.parameter, &rel.covariate).is_some() {
        return Err(format!(
            "ferx-core::edit: [covariate_model] already declares `{} ~ {}` — one line per \
             (parameter, covariate) pair. Drop it first if you mean to change its form.",
            rel.parameter, rel.covariate
        ));
    }
    text.append_to_block("covariate_model", &rel.render());
    Ok(())
}

fn drop_covariate_relation(text: &mut ModelText, param: &str, cov: &str) -> Result<(), String> {
    let Some(line) = find_relation_line(text, param, cov) else {
        return Err(format!(
            "ferx-core::edit: [covariate_model] declares no `{param} ~ {cov}` relation to drop"
        ));
    };
    text.delete_lines(vec![line]);
    // An explicit `=> THETA(...)` clause may have deferred to a θ declared in
    // [parameters]; that θ is now unreferenced and would fail the parser's
    // "declared but never used" check.
    prune_unreferenced(text)
}

/// The `[covariate_model]` line for one `(parameter, covariate)` pair.
///
/// Reads only as far as the pair — the form and the `=> …` clause are the
/// parser's business, and a relation this cannot address is one the parser
/// would reject anyway.
fn find_relation_line(text: &ModelText, param: &str, cov: &str) -> Option<usize> {
    find_line(text, "covariate_model", |code| {
        let body = code.split("=>").next().unwrap_or(code);
        let Some((p, rest)) = body.split_once('~') else {
            return false;
        };
        p.trim() == param && rest.trim().split_whitespace().next() == Some(cov)
    })
}

// ── η surgery ───────────────────────────────────────────────────────────────

fn add_iiv(text: &mut ModelText, param: &str, form: &IivForm) -> Result<(), String> {
    finite(form.variance(), &format!("`omega {}` variance", form.eta()))?;
    let Some((line, rhs)) = find_assignment(text, param) else {
        return Err(format!(
            "ferx-core::edit: [individual_parameters] declares no `{param}` to give an η to"
        ));
    };
    if omega_decl_line(text, form.eta()).is_some() {
        return Err(format!(
            "ferx-core::edit: `omega {}` is already declared",
            form.eta()
        ));
    }
    // `OMEGA_RE` is anchored on `omega`, so it does not see an η that a
    // `block_omega` declares — appending a diagonal `omega` for one would
    // declare the same random effect twice.
    if let Some(block) = block_omega_line_for(text, form.eta()) {
        return Err(format!(
            "ferx-core::edit: `{}` is already declared by `{}`",
            form.eta(),
            text.code(block)
        ));
    }
    if let Some(existing) = declared_etas(text).iter().find(|e| mentions(&rhs, e)) {
        return Err(format!(
            "ferx-core::edit: `{param}` already carries the random effect `{existing}` — a \
             parameter gets one η here. Drop it first if you mean to replace it."
        ));
    }
    let new_rhs = match form {
        IivForm::Exponential { eta, .. } => {
            require_product(param, &rhs)?;
            format!("{rhs} * exp({eta})")
        }
        IivForm::Proportional { eta, .. } => {
            require_product(param, &rhs)?;
            format!("{rhs} * (1 + {eta})")
        }
        IivForm::Additive { eta, .. } => format!("{rhs} + {eta}"),
    };

    text.set_code(line, &format!("{param} = {new_rhs}"));
    text.append_to_block(
        "parameters",
        &format!("omega {} ~ {}", form.eta(), num(form.variance())),
    );
    Ok(())
}

fn drop_iiv(text: &mut ModelText, param: &str) -> Result<(), String> {
    let Some((line, rhs)) = find_assignment(text, param) else {
        return Err(format!(
            "ferx-core::edit: [individual_parameters] declares no `{param}` to take an η from"
        ));
    };
    let etas = declared_etas(text);
    // The canonical form: the last top-level factor is `exp(ETA_x)` and
    // nothing else on the line mentions an η.
    let (head, eta) = split_trailing_exp_eta(&rhs, &etas).ok_or_else(|| {
        format!(
            "ferx-core::edit: `{param} = {rhs}` is not in the canonical form \
             `{param} = TVP * exp(ETA_{param})`, so its η cannot be removed by rewriting the \
             line. Rewrite `{param}` as a product ending in `exp(<eta>)`, or edit \
             [individual_parameters] by hand."
        )
    })?;
    if etas.iter().any(|e| e != &eta && mentions(&head, e)) {
        return Err(format!(
            "ferx-core::edit: `{param} = {rhs}` mentions more than one random effect; dropping \
             one by rewriting the line would be a guess about which."
        ));
    }
    if let Some(block) = block_omega_line_for(text, &eta) {
        return Err(format!(
            "ferx-core::edit: `{eta}` is part of `{}` — a block_omega cannot have one η removed \
             from it by line surgery. Redeclare the block without `{eta}` first.",
            text.code(block)
        ));
    }
    let Some(omega) = omega_decl_line(text, &eta) else {
        return Err(format!(
            "ferx-core::edit: `{param}` uses `{eta}`, which has no `omega {eta} ~ …` declaration"
        ));
    };

    text.set_code(line, &format!("{param} = {head}"));
    text.delete_lines(vec![omega]);
    Ok(())
}

/// Split `… * exp(ETA)` into (`…`, `ETA`) when the right-hand side ends in a
/// top-level `exp` of a declared η, which is the canonical IIV form.
fn split_trailing_exp_eta(rhs: &str, etas: &[String]) -> Option<(String, String)> {
    let rhs = rhs.trim();
    let factors = top_level_factors(rhs)?;
    let (last, head) = factors.split_last()?;
    let inner = last
        .trim()
        .strip_prefix("exp")?
        .trim()
        .strip_prefix('(')?
        .strip_suffix(')')?;
    let eta = inner.trim();
    if !etas.iter().any(|e| e == eta) {
        return None;
    }
    if head.is_empty() {
        return None;
    }
    Some((head.join(" * "), eta.to_string()))
}

/// Split a right-hand side into its top-level `*` factors, or `None` when it
/// is not a product at all (a sum, a conditional, a transform).
fn top_level_factors(rhs: &str) -> Option<Vec<String>> {
    if !is_top_level_product(rhs) {
        return None;
    }
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in rhs.chars() {
        match c {
            '(' | '[' => {
                depth += 1;
                cur.push(c);
            }
            ')' | ']' => {
                depth -= 1;
                cur.push(c);
            }
            '*' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    out.push(cur.trim().to_string());
    if out.iter().any(|f| f.is_empty()) {
        return None;
    }
    Some(out)
}

/// Whether `rhs` is a product of factors at the top level — no `+`, `-`, `/`
/// or conditional outside parentheses.
///
/// A covariate factor or an `exp(η)` appended to something that is *not* a
/// top-level product changes the arithmetic silently (`A + B * exp(η)` is not
/// `(A + B) * exp(η)`), which is why this is a precondition rather than a
/// best-effort rewrite — the same posture `[covariate_model]` takes (#1111).
/// `/` joins the list because `A / B * exp(η)` reassociates too.
fn is_top_level_product(rhs: &str) -> bool {
    let bytes: Vec<char> = rhs.chars().collect();
    let mut depth = 0i32;
    for (i, c) in bytes.iter().enumerate() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            '+' | '-' if depth == 0 => {
                // A sign inside a float exponent (`1e-5`) is part of the
                // number. One character of context is not enough to tell that
                // from an identifier that merely ends in `e`: `BASE-1`,
                // `AGE-40` and `SIZE-1` all have an `e` in front of the sign.
                // A number has to be there too, on both sides of the `e`.
                let exponent = i
                    .checked_sub(1)
                    .is_some_and(|p| matches!(bytes[p], 'e' | 'E'))
                    && i.checked_sub(2)
                        .is_some_and(|p| bytes[p].is_ascii_digit() || bytes[p] == '.')
                    && bytes.get(i + 1).is_some_and(|c| c.is_ascii_digit());
                if !exponent {
                    return false;
                }
            }
            '/' | '?' | ':' | ',' if depth == 0 => return false,
            _ => {}
        }
    }
    depth == 0
}

fn require_product(param: &str, rhs: &str) -> Result<(), String> {
    if is_top_level_product(rhs) {
        Ok(())
    } else {
        Err(format!(
            "ferx-core::edit: `{param} = {rhs}` is not a top-level product, so a multiplicative \
             η factor cannot be appended without changing the arithmetic. Write `{param}` in the \
             canonical form `TVP * …`, or add the η by hand."
        ))
    }
}

// ── block_omega ─────────────────────────────────────────────────────────────

/// Correlation seeded into the off-diagonal of a freshly blocked omega.
///
/// Not zero: a search blocks two ηs precisely to estimate their correlation,
/// and starting the Cholesky factor on the diagonal makes the first step a
/// less informative one. 0.1 is the same weak-but-present seed Pharmpy's
/// `create_joint_distribution` uses.
const SEED_CORRELATION: f64 = 0.1;

fn set_omega_block(text: &mut ModelText, names: &[String]) -> Result<(), String> {
    if names.len() < 2 {
        return Err(
            "ferx-core::edit: SetOmegaBlock needs at least two η — a 1×1 block is the diagonal \
             declaration it would replace"
                .to_string(),
        );
    }
    let mut seen = std::collections::HashSet::new();
    for n in names {
        if !seen.insert(n) {
            return Err(format!(
                "ferx-core::edit: `{n}` is listed twice in the omega block"
            ));
        }
    }
    // Collected per η and then sorted by declaration line: the block takes the
    // place of the first declaration, so listing the η in *source* order — not
    // in the caller's argument order — is what keeps each one's index in the
    // omega matrix, and therefore the meaning of every `eta_names` position, as
    // it was before the block.
    let mut slots: Vec<(usize, &str, f64)> = Vec::with_capacity(names.len());
    for n in names {
        if let Some(b) = block_omega_line_for(text, n) {
            return Err(format!(
                "ferx-core::edit: `{n}` is already blocked by `{}`",
                text.code(b)
            ));
        }
        let Some(i) = omega_decl_line(text, n) else {
            return Err(format!(
                "ferx-core::edit: [parameters] declares no `omega {n} ~ …` to block"
            ));
        };
        slots.push((i, n.as_str(), omega_variance(text.code(i))?));
    }
    slots.sort_by_key(|(i, _, _)| *i);
    let lines: Vec<usize> = slots.iter().map(|(i, _, _)| *i).collect();
    let ordered: Vec<&str> = slots.iter().map(|(_, n, _)| *n).collect();
    let variances: Vec<f64> = slots.iter().map(|(_, _, v)| *v).collect();

    let mut tri = Vec::new();
    for (i, vi) in variances.iter().enumerate() {
        for vj in variances.iter().take(i) {
            tri.push(SEED_CORRELATION * (vi * vj).sqrt());
        }
        tri.push(*vi);
    }
    let values: Vec<String> = tri.iter().map(|v| num(*v)).collect();
    let decl = format!(
        "block_omega ({}) = [{}]",
        ordered.join(", "),
        values.join(", ")
    );

    let first = *lines.iter().min().expect("names is non-empty");
    text.set_code(first, &decl);
    text.delete_lines(lines.into_iter().filter(|i| *i != first).collect());
    Ok(())
}

/// The variance an `omega NAME ~ v [(sd)]` line declares. `block_omega` is
/// variance-scale only, so an `(sd)` declaration is squared on the way in.
fn omega_variance(code: &str) -> Result<f64, String> {
    let caps = OMEGA_RE
        .captures(code)
        .ok_or_else(|| format!("ferx-core::edit: cannot read the variance of `{code}`"))?;
    let raw: f64 = caps[3].parse().map_err(|_| {
        format!(
            "ferx-core::edit: `{}` is not a number in `{code}`",
            &caps[3]
        )
    })?;
    Ok(if SD_RE.is_match(code) { raw * raw } else { raw })
}

// ── [error_model] ───────────────────────────────────────────────────────────

fn set_error_model(text: &mut ModelText, spec: &ErrorSpecText) -> Result<(), String> {
    if spec.sigmas.len() != spec.form.n_sigma() {
        return Err(format!(
            "ferx-core::edit: a {} error model takes {} sigma(s), {} given",
            spec.form.label(),
            spec.form.n_sigma(),
            spec.sigmas.len()
        ));
    }
    for s in &spec.sigmas {
        finite(s.init, &format!("`sigma {}` init", s.name))?;
    }
    // σ are reconciled positionally against the `sigma NAME ~ …` declarations
    // below, and `SIGMA_RE` does not match a `block_sigma` line — so a
    // correlated σ block would have the new σ *appended* beside it rather than
    // reconciled with it, leaving the model with more σ than the error model
    // names.
    if let Some(i) = find_line(text, "parameters", |c| BLOCK_SIGMA_RE.is_match(c)) {
        return Err(format!(
            "ferx-core::edit: [parameters] declares σ as `{}`. SetErrorModel lays σ out \
             positionally and cannot reconcile a correlated σ block; edit [parameters] by hand.",
            text.code(i)
        ));
    }
    let existing: Vec<usize> = text
        .body_indices("error_model")
        .into_iter()
        .filter(|i| !text.code(*i).is_empty())
        .collect();
    // A per-CMT or covariate-selected block describes endpoints this edit does
    // not know about; replacing it wholesale would drop them silently.
    if existing.len() > 1
        || existing
            .iter()
            .any(|i| text.code(*i).contains("CMT") || text.code(*i).starts_with("if"))
    {
        return Err(
            "ferx-core::edit: [error_model] is a per-CMT or covariate-selected block. \
             SetErrorModel replaces a single `DV ~ TYPE(...)` statement and would drop the \
             other endpoints; edit the block by hand."
                .to_string(),
        );
    }

    let names: Vec<&str> = spec.sigmas.iter().map(|s| s.name.as_str()).collect();
    let stmt = format!(
        "{} ~ {}({})",
        spec.endpoint,
        spec.form.label(),
        names.join(", ")
    );
    match existing.first() {
        Some(i) => text.set_code(*i, &stmt),
        None => text.append_to_block("error_model", &stmt),
    }

    // A single-endpoint `[error_model]` consumes its σ **positionally** from
    // the `[parameters]` declaration order — the names in `combined(a, b)` are
    // checked against that order, not resolved by it. So appending a new σ at
    // the end of the block is not enough: the declarations have to be laid out
    // in the order the statement names them, or the model does not parse (and,
    // for `combined`, a spelling that did parse would have swapped which σ is
    // the proportional component).
    let slots: Vec<usize> = text
        .body_indices("parameters")
        .into_iter()
        .filter(|i| SIGMA_RE.is_match(text.code(*i)))
        .collect();
    let wanted: Vec<String> = spec
        .sigmas
        .iter()
        .map(|s| {
            // A σ the model already declares keeps its own line verbatim — its
            // init, its scale annotation and its `FIX` are the author's, and
            // `SetErrorModel` is not the edit that changes them.
            slots
                .iter()
                .map(|i| text.code(*i).to_string())
                .find(|c| SIGMA_RE.captures(c).is_some_and(|m| m[2] == s.name))
                .unwrap_or_else(|| s.render())
        })
        .collect();
    for (slot, code) in slots.iter().zip(&wanted) {
        text.set_code(*slot, code);
    }
    // σ the previous error model referenced and this one does not are dead.
    // The parser does not reject an unused σ, so leaving one behind would have
    // the candidate report a parameter it never estimates.
    text.delete_lines(slots.iter().skip(wanted.len()).copied().collect());
    for code in wanted.iter().skip(slots.len()) {
        text.append_to_block("parameters", code);
    }
    Ok(())
}

// ── Initial estimates ───────────────────────────────────────────────────────

fn seed_inits(text: &mut ModelText, fit: &FitResult) -> Result<(), String> {
    let theta: HashMap<&str, f64> = fit
        .theta_names
        .iter()
        .map(|s| s.as_str())
        .zip(fit.theta.iter().copied())
        .collect();
    let eta_index: HashMap<&str, usize> = fit
        .eta_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let sigma: HashMap<&str, f64> = fit
        .sigma_names
        .iter()
        .map(|s| s.as_str())
        .zip(fit.sigma.iter().copied())
        .collect();

    // A `FIX`ed parameter is a statement about the model, not an estimate to
    // carry over; a vector θ (`theta NAME[...]`) has no single init to write.
    for i in text.body_indices("parameters") {
        let code = text.code(i).to_string();
        if FIX_RE.is_match(&code) {
            continue;
        }
        if let Some(caps) = THETA_RE.captures(&code) {
            // A vector θ (`theta NAME[...](…)`) does not match at all — the
            // regex wants `(` straight after the name — so it is skipped here
            // for free, which is right: it has no single init to write.
            if let Some(v) = theta.get(&caps[2]).copied().filter(|v| v.is_finite()) {
                let new = format!("{}{}{}", &caps[1], num(v), &caps[4]);
                text.set_code(i, &new);
            }
        } else if let Some(caps) = OMEGA_RE.captures(&code) {
            let Some(k) = eta_index.get(&caps[2]).copied() else {
                continue;
            };
            let var = fit.omega[(k, k)];
            if !var.is_finite() {
                continue;
            }
            // The declaration's own scale wins: rewriting `(sd)` as a variance
            // would change the number's meaning as well as its value.
            let value = if SD_RE.is_match(&code) {
                var.sqrt()
            } else {
                var
            };
            text.set_code(i, &format!("{}{}{}", &caps[1], num(value), &caps[4]));
        } else if let Some(caps) = SIGMA_RE.captures(&code) {
            // `FitResult::sigma` is on the SD scale throughout the engine.
            let Some(sd) = sigma.get(&caps[2]).copied().filter(|v| v.is_finite()) else {
                continue;
            };
            let value = if SD_RE.is_match(&code) { sd } else { sd * sd };
            text.set_code(i, &format!("{}{}{}", &caps[1], num(value), &caps[4]));
        } else if let Some(caps) = BLOCK_OMEGA_RE.captures(&code) {
            let names: Vec<String> = caps[2]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let idx: Option<Vec<usize>> = names
                .iter()
                .map(|n| eta_index.get(n.as_str()).copied())
                .collect();
            let Some(idx) = idx else { continue };
            let mut tri = Vec::new();
            for (a, ia) in idx.iter().enumerate() {
                for ib in idx.iter().take(a + 1) {
                    tri.push(fit.omega[(*ia, *ib)]);
                }
            }
            if tri.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let values: Vec<String> = tri.iter().map(|v| num(*v)).collect();
            text.set_code(
                i,
                &format!("{}[{}]{}", &caps[1], values.join(", "), &caps[4]),
            );
        }
    }
    Ok(())
}

// ── [fit_options] ───────────────────────────────────────────────────────────

fn set_fit_option(text: &mut ModelText, key: &str, value: &str) -> Result<(), String> {
    if key.trim().is_empty() {
        return Err("ferx-core::edit: SetFitOption needs a key".to_string());
    }
    if value.contains('\n') || key.contains('\n') {
        return Err(format!(
            "ferx-core::edit: `{key} = {value}` spans a line break; a fit option is one line"
        ));
    }
    // `code_of` strips `#` and `//` before the parser ever sees the line, so a
    // marker anywhere in the option would silently truncate it on read-back.
    if [key, value]
        .iter()
        .any(|s| s.contains('#') || s.contains("//"))
    {
        return Err(format!(
            "ferx-core::edit: `{key} = {value}` contains a comment marker (`#` or `//`), which \
             is stripped when the line is read back — the option would be truncated"
        ));
    }
    match find_line(text, "fit_options", |c| {
        c.split_once('=').is_some_and(|(k, _)| k.trim() == key)
    }) {
        Some(i) => {
            // Keep the author's column alignment: only the value is replaced.
            let code = text.code(i).to_string();
            let (lhs, _) = code.split_once('=').expect("matched on the `=`");
            text.set_code(i, &format!("{lhs}= {}", value.trim()));
        }
        None => text.append_to_block("fit_options", &format!("{key} = {}", value.trim())),
    }
    Ok(())
}

// ── Shared lookups ──────────────────────────────────────────────────────────

fn find_line(text: &ModelText, block: &str, pred: impl Fn(&str) -> bool) -> Option<usize> {
    text.body_indices(block)
        .into_iter()
        .find(|i| pred(text.code(*i)))
}

/// `[individual_parameters]` names in source order, mapped to their line.
fn individual_parameter_names(text: &ModelText) -> HashMap<String, usize> {
    let mut out = HashMap::new();
    for i in text.body_indices("individual_parameters") {
        if let Some(caps) = ASSIGN_RE.captures(text.code(i)) {
            out.entry(caps[1].to_string()).or_insert(i);
        }
    }
    out
}

fn find_assignment(text: &ModelText, param: &str) -> Option<(usize, String)> {
    text.body_indices("individual_parameters")
        .into_iter()
        .find_map(|i| {
            let caps = ASSIGN_RE.captures(text.code(i))?;
            (&caps[1] == param).then(|| (i, caps[2].trim().to_string()))
        })
}

fn omega_decl_line(text: &ModelText, eta: &str) -> Option<usize> {
    find_line(text, "parameters", |c| {
        OMEGA_RE.captures(c).is_some_and(|m| &m[2] == eta)
    })
}

fn block_omega_line_for(text: &ModelText, eta: &str) -> Option<usize> {
    find_line(text, "parameters", |c| {
        BLOCK_OMEGA_RE
            .captures(c)
            .is_some_and(|m| m[2].split(',').any(|n| n.trim() == eta))
    })
}

/// Every η the model declares — diagonal `omega` and `block_omega` alike.
fn declared_etas(text: &ModelText) -> Vec<String> {
    let mut out = Vec::new();
    for i in text.body_indices("parameters") {
        let code = text.code(i);
        if let Some(caps) = OMEGA_RE.captures(code) {
            out.push(caps[2].to_string());
        } else if let Some(caps) = BLOCK_OMEGA_RE.captures(code) {
            out.extend(
                caps[2]
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty()),
            );
        }
    }
    out
}

/// Whether `expr` uses `name` as a whole identifier.
fn mentions(expr: &str, name: &str) -> bool {
    super::IDENT_RE.find_iter(expr).any(|m| m.as_str() == name)
}

// ── Declaration pruning ─────────────────────────────────────────────────────

/// Delete `theta` / `omega` declarations and `[individual_parameters]`
/// assignments that nothing references any more, to a fixed point.
///
/// This is what makes a structural swap a single edit: `two_cpt_oral →
/// one_cpt_oral` drops `q=Q, v2=V2` from the `pk` line, which orphans `Q` and
/// `V2`, which orphans `TVQ`, `TVV2`, `ETA_Q` and `ETA_V2` — four blocks' worth
/// of coupled deletions that the caller would otherwise have to enumerate, and
/// which the parser rejects (`computed but never used`) if any is missed.
///
/// Reachability is textual and therefore conservative. It is a *closure*, not
/// a per-line occurrence count, because a declaration and the thing that would
/// keep it alive can hold each other up: `Q = TVQ` mentions `Q`, and a
/// `[covariate_model]` line `Q ~ WT power(...)` mentions it again, so counting
/// occurrences would keep an orphaned `Q` forever. Instead the live set starts
/// at the blocks that *consume* parameters — `[structural_model]`, `[odes]`,
/// `[scaling]`, `[error_model]`, everything that is not itself a declaration —
/// and grows through the declarations those reach.
///
/// # Errors
///
/// When something is dead that this cannot remove safely: an assignment inside
/// an `if`/`else` branch, or a `block_omega` whose triangle it cannot read.
/// Leaving one behind is not the conservative option — an unused η is not a
/// parse error, it is a flat direction in the omega matrix and an extra
/// `n_parameters`, which is a wrong BIC for the search that ranks on it.
fn prune_unreferenced(text: &mut ModelText) -> Result<(), String> {
    // An `if`/`else` in [individual_parameters] puts an assignment inside a
    // branch, where "this line declares this name" stops being true. The
    // assignments are still read as declarations — treating them as roots
    // instead would keep every θ and η they mention alive, which is the
    // orphaned-`TVQ`-and-phantom-η case — but a *dead* one is refused rather
    // than deleted, because removing a line from inside a branch is a guess
    // about the branch.
    let individual = text.body_indices("individual_parameters");
    let conditional = individual.iter().any(|i| {
        let c = text.code(*i);
        c.starts_with("if") || c.starts_with("else") || c.contains('{') || c.contains('}')
    });

    // Declaration lines, by what they declare. Everything else in the file is
    // a consumer and therefore a root.
    let mut decls: Vec<(String, usize)> = Vec::new();
    let mut relations: Vec<(String, usize)> = Vec::new();
    // `block_omega (A, B) = […]` declares several η on one line, and declares
    // them nowhere else — so it has to be excluded from the root scan like any
    // other declaration, or each η it names would keep *itself* alive.
    let mut blocks: Vec<(Vec<String>, usize)> = Vec::new();
    for i in text.body_indices("parameters") {
        let code = text.code(i);
        if let Some(caps) = THETA_RE.captures(code) {
            decls.push((caps[2].to_string(), i));
        } else if let Some(caps) = OMEGA_RE.captures(code) {
            decls.push((caps[2].to_string(), i));
        } else if let Some(caps) = BLOCK_OMEGA_RE.captures(code) {
            let names: Vec<String> = caps[2]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !names.is_empty() {
                blocks.push((names, i));
            }
        }
    }
    for i in &individual {
        if let Some(caps) = ASSIGN_RE.captures(text.code(*i)) {
            decls.push((caps[1].to_string(), *i));
        }
    }
    for i in text.body_indices("covariate_model") {
        let code = text.code(i);
        if code.is_empty() {
            continue;
        }
        let body = code.split("=>").next().unwrap_or(code);
        if let Some((p, _)) = body.split_once('~') {
            relations.push((p.trim().to_string(), i));
        }
    }

    let mut declaration_lines: Vec<usize> = decls.iter().map(|(_, i)| *i).collect();
    declaration_lines.extend(relations.iter().map(|(_, i)| *i));
    declaration_lines.extend(blocks.iter().map(|(_, i)| *i));
    let mut live = text.identifiers_excluding(&declaration_lines);

    // Grow the live set: a live name pulls in whatever its declaration reads.
    loop {
        let before = live.len();
        for (name, i) in decls.iter().chain(relations.iter()) {
            if !live.contains(name) {
                continue;
            }
            for m in super::IDENT_RE.find_iter(text.code(*i)) {
                live.insert(m.as_str().to_string());
            }
        }
        // A relation's θ is *generated* — `THETA_<PARAM>_<COV>` and its
        // `_LO`/`_HI`/`_<LEVEL>` variants never appear in the block text — so a
        // `[parameters]` declaration the desugar would defer to has no textual
        // reference to keep it alive. Protect the whole family by prefix.
        for (param, i) in &relations {
            if !live.contains(param) {
                continue;
            }
            let code = text.code(*i);
            let body = code.split("=>").next().unwrap_or(code);
            let Some((_, rest)) = body.split_once('~') else {
                continue;
            };
            let Some(cov) = rest.trim().split_whitespace().next() else {
                continue;
            };
            let prefix = format!("THETA_{param}_{cov}");
            let generated: Vec<String> = decls
                .iter()
                .map(|(n, _)| n.clone())
                .filter(|n| n.starts_with(&prefix))
                .collect();
            live.extend(generated);
        }
        if live.len() == before {
            break;
        }
    }

    let dead: Vec<(&str, usize)> = decls
        .iter()
        .chain(relations.iter())
        .filter(|(name, _)| !live.contains(name))
        .map(|(name, i)| (name.as_str(), *i))
        .collect();
    if conditional {
        if let Some((name, _)) = dead.iter().find(|(_, i)| individual.contains(i)) {
            return Err(format!(
                "ferx-core::edit: `{name}` in [individual_parameters] is no longer referenced, \
                 but the block contains an `if`/`else`, so removing a line from it would be a \
                 guess about the branch. Edit [individual_parameters] by hand."
            ));
        }
    }
    let mut dead_lines: Vec<usize> = dead.into_iter().map(|(_, i)| i).collect();

    // A `block_omega` that lost an η is rewritten around the ones that are
    // left, rather than kept whole (a random effect no parameter uses) or
    // deleted whole (which would take the live η with it).
    for (names, i) in &blocks {
        let keep: Vec<usize> = (0..names.len())
            .filter(|k| live.contains(&names[*k]))
            .collect();
        if keep.len() == names.len() {
            continue;
        }
        if keep.is_empty() {
            dead_lines.push(*i);
            continue;
        }
        let decl = shrink_omega_block(text.code(*i), names, &keep)?;
        text.set_code(*i, &decl);
    }

    text.delete_lines(dead_lines);
    Ok(())
}

/// Rewrite `block_omega (A, B, C) = […]` around the η at the indices in
/// `keep`, which are ascending.
///
/// The triangle is row-major lower-triangular, so entry (a, b) with a ≥ b sits
/// at `a(a+1)/2 + b` and the kept sub-block is that lookup over the kept
/// indices — the surviving variances and their covariances, unchanged. A lone
/// survivor is written back as the diagonal `omega` it would otherwise be a
/// 1×1 block of, which is not a form the parser accepts.
fn shrink_omega_block(code: &str, names: &[String], keep: &[usize]) -> Result<String, String> {
    let caps = BLOCK_OMEGA_RE
        .captures(code)
        .ok_or_else(|| format!("ferx-core::edit: cannot read `{code}`"))?;
    let values: Vec<f64> = caps[3]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<f64>()
                .map_err(|_| format!("ferx-core::edit: `{s}` is not a number in `{code}`"))
        })
        .collect::<Result<_, _>>()?;
    let n = names.len();
    if values.len() != n * (n + 1) / 2 {
        return Err(format!(
            "ferx-core::edit: `{code}` names {n} η but declares {} lower-triangle values, so the \
             random effect it holds that nothing uses any more cannot be removed from it. \
             Redeclare the block by hand.",
            values.len()
        ));
    }
    let rest = &caps[4];
    if let [k] = keep {
        let var = values[k * (k + 1) / 2 + k];
        return Ok(format!("omega {} ~ {}{rest}", names[*k], num(var)));
    }
    let mut tri = Vec::with_capacity(keep.len() * (keep.len() + 1) / 2);
    for (a, ia) in keep.iter().enumerate() {
        for ib in keep.iter().take(a + 1) {
            tri.push(num(values[ia * (ia + 1) / 2 + ib]));
        }
    }
    let kept: Vec<&str> = keep.iter().map(|k| names[*k].as_str()).collect();
    Ok(format!(
        "block_omega ({}) = [{}]{rest}",
        kept.join(", "),
        tri.join(", ")
    ))
}
