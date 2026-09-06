//! The [`ModelEdit`] appliers.
//!
//! Every variant follows the same shape: **validate everything first, mutate
//! second**. A rejected edit must leave the text exactly as it found it, or a
//! search loop that recovers from a bad proposal would carry a half-written
//! model into the next candidate.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::LazyLock;

use regex::Regex;

use super::spec::{
    finite, num, ErrorForm, ErrorSpecText, EtaDecl, IivForm, ModelEdit, Relation, SigmaDecl,
    StructuralSpec, ThetaDecl, TimeVaryingDecl,
};
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
/// `iiv_on_ruv = ETA` in `[error_model]` — group 1 the η.
static IIV_ON_RUV_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^iiv_on_ruv\s*=\s*(\w+)\s*$").unwrap());
/// `DV ~ form(args)` — the single-endpoint statement `SetErrorModel` writes.
static ERROR_STMT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\w+)\s*~\s*(\w+)\s*\((.+)\)\s*$").unwrap());
/// The canonical time-varying σ argument `σ * (if (TAD < c) θ else 1.0)` —
/// groups: σ, cutoff, θ.
static TIME_VARYING_ARG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(\w+)\s*\*\s*\(\s*if\s*\(\s*TAD\s*<\s*([0-9eE.+-]+)\s*\)\s*(\w+)\s+else\s+1(?:\.0*)?\s*\)$",
    )
    .unwrap()
});
/// A bare identifier and nothing else.
static BARE_IDENT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Za-z_]\w*$").unwrap());
/// The two bounds after a θ init: `, lower, upper`.
static THETA_BOUNDS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*,\s*([0-9eE.+-]+)\s*,\s*([0-9eE.+-]+)").unwrap());

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

    let Some((pk_span, _)) = find_decl(text, "structural_model", |c| PK_RE.is_match(c)) else {
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
        if existing.contains(var) {
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
    let stale = text.set_span(
        &pk_span,
        &format!("pk {}({})", spec.template, pairs.join(", ")),
    );
    text.delete_lines(stale);

    for p in to_create {
        let rhs = match &p.iiv {
            Some((eta, _)) => format!("{} * exp({eta})", p.theta),
            None => p.theta.clone(),
        };
        text.append_to_block("individual_parameters", &format!("{} = {rhs}", p.name));
        text.append_to_block(
            "parameters",
            &format!(
                "theta {}({}, {}, {}){}",
                p.theta,
                num(p.init),
                num(p.lower),
                num(p.upper),
                if p.fixed { " FIX" } else { "" }
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
    if find_relation_decl(text, &rel.parameter, &rel.covariate).is_some() {
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
    let Some(span) = find_relation_decl(text, param, cov) else {
        return Err(format!(
            "ferx-core::edit: [covariate_model] declares no `{param} ~ {cov}` relation to drop"
        ));
    };
    text.delete_lines(span.collect());
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
fn find_relation_decl(text: &ModelText, param: &str, cov: &str) -> Option<Range<usize>> {
    find_decl(text, "covariate_model", |code| {
        let body = code.split("=>").next().unwrap_or(code);
        let Some((p, rest)) = body.split_once('~') else {
            return false;
        };
        p.trim() == param && rest.trim().split_whitespace().next() == Some(cov)
    })
    .map(|(span, _)| span)
}

// ── η surgery ───────────────────────────────────────────────────────────────

fn add_iiv(text: &mut ModelText, param: &str, form: &IivForm) -> Result<(), String> {
    finite(form.variance(), &format!("`omega {}` variance", form.eta()))?;
    let Some((span, rhs)) = find_assignment(text, param) else {
        return Err(format!(
            "ferx-core::edit: [individual_parameters] declares no `{param}` to give an η to"
        ));
    };
    if omega_decl(text, form.eta()).is_some() {
        return Err(format!(
            "ferx-core::edit: `omega {}` is already declared",
            form.eta()
        ));
    }
    // `OMEGA_RE` is anchored on `omega`, so it does not see an η that a
    // `block_omega` declares — appending a diagonal `omega` for one would
    // declare the same random effect twice.
    if let Some((_, block)) = block_omega_decl(text, form.eta()) {
        return Err(format!(
            "ferx-core::edit: `{}` is already declared by `{block}`",
            form.eta()
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

    let stale = text.set_span(&span, &format!("{param} = {new_rhs}"));
    text.delete_lines(stale);
    text.append_to_block(
        "parameters",
        &format!("omega {} ~ {}", form.eta(), num(form.variance())),
    );
    Ok(())
}

fn drop_iiv(text: &mut ModelText, param: &str) -> Result<(), String> {
    let Some((span, rhs)) = find_assignment(text, param) else {
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
    if let Some((_, block)) = block_omega_decl(text, &eta) {
        return Err(format!(
            "ferx-core::edit: `{eta}` is part of `{block}` — a block_omega cannot have one η \
             removed from it by line surgery. Redeclare the block without `{eta}` first."
        ));
    }
    let Some((omega, _)) = omega_decl(text, &eta) else {
        return Err(format!(
            "ferx-core::edit: `{param}` uses `{eta}`, which has no `omega {eta} ~ …` declaration"
        ));
    };

    let mut stale = text.set_span(&span, &format!("{param} = {head}"));
    // The declaration goes only if nothing else still reads the η. Two
    // parameters may legally share one — `CL = TVCL * exp(ETA_SIZE)` beside
    // `V = TVV * exp(ETA_SIZE)` — and deleting it on the first `DropIiv` would
    // leave the other expression with a random effect the model never declares.
    // The check runs on the rewritten text, and skips the lines that are on
    // their way out: they still carry the η this edit just removed.
    let mut skip: Vec<usize> = omega.clone().collect();
    skip.extend(stale.iter().copied());
    if !text.identifiers_excluding(&skip).contains(&eta) {
        stale.extend(omega);
    }
    text.delete_lines(stale);
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
    let mut slots: Vec<(Range<usize>, &str, f64, bool)> = Vec::with_capacity(names.len());
    for n in names {
        if let Some((_, block)) = block_omega_decl(text, n) {
            return Err(format!(
                "ferx-core::edit: `{n}` is already blocked by `{block}`"
            ));
        }
        let Some((span, code)) = omega_decl(text, n) else {
            return Err(format!(
                "ferx-core::edit: [parameters] declares no `omega {n} ~ …` to block"
            ));
        };
        let variance = omega_variance(&code)?;
        slots.push((span, n.as_str(), variance, FIX_RE.is_match(&code)));
    }
    slots.sort_by_key(|(span, _, _, _)| span.start);
    let ordered: Vec<&str> = slots.iter().map(|(_, n, _, _)| *n).collect();
    let variances: Vec<f64> = slots.iter().map(|(_, _, v, _)| *v).collect();
    // `FIX` is a block-level flag: one `block_omega` cannot fix some of its
    // entries and estimate the others. Dropping it silently would turn fixed
    // covariance parameters into estimated ones — a different model, and one
    // whose extra parameters the search would not know it had added.
    let fixed = slots.iter().filter(|(_, _, _, f)| *f).count();
    if fixed != 0 && fixed != slots.len() {
        let free: Vec<&str> = slots
            .iter()
            .filter(|(_, _, _, f)| !f)
            .map(|(_, n, _, _)| *n)
            .collect();
        return Err(format!(
            "ferx-core::edit: {} of the η to block are `FIX`ed and {} are not ({}). `FIX` is a \
             block-level flag, so a mixed block cannot be written; fix or free them all first.",
            fixed,
            free.len(),
            free.join(", ")
        ));
    }

    let mut tri = Vec::new();
    for (i, vi) in variances.iter().enumerate() {
        for vj in variances.iter().take(i) {
            tri.push(SEED_CORRELATION * (vi * vj).sqrt());
        }
        tri.push(*vi);
    }
    let values: Vec<String> = tri.iter().map(|v| num(*v)).collect();
    let decl = format!(
        "block_omega ({}) = [{}]{}",
        ordered.join(", "),
        values.join(", "),
        if fixed > 0 { " FIX" } else { "" }
    );

    // `slots` is in source order, so the first declaration is the one the
    // block takes the place of.
    let (first, rest) = slots.split_first().expect("names is non-empty");
    let mut stale = text.set_span(&first.0, &decl);
    for (span, ..) in rest {
        stale.extend(span.clone());
    }
    text.delete_lines(stale);
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
    match (spec.form, &spec.exponent) {
        (ErrorForm::Power, None) => {
            return Err("ferx-core::edit: a power error model needs its exponent θ \
                 (`ErrorSpecText::with_exponent`)"
                .to_string())
        }
        (form, Some(p)) if form != ErrorForm::Power => {
            return Err(format!(
                "ferx-core::edit: an exponent θ `{}` belongs to a power error model, not to {}",
                p.name,
                form.label()
            ))
        }
        _ => {}
    }
    for t in spec
        .exponent
        .iter()
        .chain(spec.time_varying.as_ref().map(|tv| &tv.theta))
    {
        finite(t.init, &format!("`theta {}` init", t.name))?;
        // An infinite bound is "no bound"; only `NaN` cannot be written.
        for (b, what) in [(t.lower, "lower"), (t.upper, "upper")] {
            if b.is_nan() {
                return Err(format!(
                    "ferx-core::edit: `theta {}` {what} bound is NaN, which cannot be written \
                     as a `.ferx` number",
                    t.name
                ));
            }
        }
        if t.lower.is_finite() != t.upper.is_finite() {
            return Err(format!(
                "ferx-core::edit: `theta {}` has one finite and one infinite bound; a `.ferx` \
                 declaration takes both or neither",
                t.name
            ));
        }
    }
    if let Some(tv) = &spec.time_varying {
        finite(tv.cutoff, "the time-varying `TAD` cutoff")?;
    }
    if let Some(eta) = &spec.iiv_on_ruv {
        finite(eta.variance, &format!("`omega {}` variance", eta.name))?;
    }
    // σ are reconciled positionally against the `sigma NAME ~ …` declarations
    // below, and `SIGMA_RE` does not match a `block_sigma` line — so a
    // correlated σ block would have the new σ *appended* beside it rather than
    // reconciled with it, leaving the model with more σ than the error model
    // names.
    if let Some((_, block)) = find_decl(text, "parameters", |c| BLOCK_SIGMA_RE.is_match(c)) {
        return Err(format!(
            "ferx-core::edit: [parameters] declares σ as `{block}`. SetErrorModel lays σ out \
             positionally and cannot reconcile a correlated σ block; edit [parameters] by hand."
        ));
    }
    // The `iiv_on_ruv = ETA` line is the spec's to set or clear; the statement
    // line(s) are what the endpoint check below looks at.
    let (iiv_lines, existing): (Vec<(Range<usize>, String)>, Vec<(Range<usize>, String)>) = text
        .logical_lines("error_model")
        .into_iter()
        .partition(|(_, c)| IIV_ON_RUV_RE.is_match(c));
    // A per-CMT or covariate-selected block describes endpoints this edit does
    // not know about; replacing it wholesale would drop them silently.
    if existing.len() > 1
        || existing
            .iter()
            .any(|(_, c)| c.contains("CMT") || c.starts_with("if"))
    {
        return Err(
            "ferx-core::edit: [error_model] is a per-CMT or covariate-selected block. \
             SetErrorModel replaces a single `DV ~ TYPE(...)` statement and would drop the \
             other endpoints; edit the block by hand."
                .to_string(),
        );
    }
    // What the old block referenced, so a θ / ω only it used can be pruned
    // once the new block is in — the exponent of a power form being replaced,
    // the η of an `iiv_on_ruv` being removed. Collected before the edit, and
    // checked for liveness after it, against the whole file.
    let old_idents: HashSet<String> = existing
        .iter()
        .chain(iiv_lines.iter())
        .flat_map(|(_, c)| super::IDENT_RE.find_iter(c).map(|m| m.as_str().to_string()))
        .collect();

    let stmt = spec.render_statement();
    match existing.first() {
        Some((span, _)) => {
            let stale = text.set_span(span, &stmt);
            text.delete_lines(stale);
        }
        None => text.append_to_block("error_model", &stmt),
    }
    // `iiv_on_ruv`: rewrite the first line, drop any duplicate, or add / remove.
    // Spans are re-read, since the statement edit above may have moved lines.
    let iiv_lines: Vec<(Range<usize>, String)> = text
        .logical_lines("error_model")
        .into_iter()
        .filter(|(_, c)| IIV_ON_RUV_RE.is_match(c))
        .collect();
    let mut stale = Vec::new();
    match (&spec.iiv_on_ruv, iiv_lines.first()) {
        (Some(eta), Some((span, _))) => {
            stale.extend(text.set_span(span, &format!("iiv_on_ruv = {}", eta.name)));
        }
        (Some(eta), None) => {
            text.append_to_block("error_model", &format!("iiv_on_ruv = {}", eta.name));
        }
        (None, _) => {}
    }
    for (span, _) in iiv_lines
        .iter()
        .skip(usize::from(spec.iiv_on_ruv.is_some()))
    {
        stale.extend(span.clone());
    }
    text.delete_lines(stale);

    // A single-endpoint `[error_model]` consumes its σ **positionally** from
    // the `[parameters]` declaration order — the names in `combined(a, b)` are
    // checked against that order, not resolved by it. So appending a new σ at
    // the end of the block is not enough: the declarations have to be laid out
    // in the order the statement names them, or the model does not parse (and,
    // for `combined`, a spelling that did parse would have swapped which σ is
    // the proportional component).
    let slots: Vec<(Range<usize>, String)> = text
        .logical_lines("parameters")
        .into_iter()
        .filter(|(_, c)| SIGMA_RE.is_match(c))
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
                .map(|(_, c)| c.clone())
                .find(|c| SIGMA_RE.captures(c).is_some_and(|m| m[2] == s.name))
                .unwrap_or_else(|| s.render())
        })
        .collect();
    let mut stale = Vec::new();
    for ((span, _), code) in slots.iter().zip(&wanted) {
        stale.extend(text.set_span(span, code));
    }
    // σ the previous error model referenced and this one does not are dead.
    // The parser does not reject an unused σ, so leaving one behind would have
    // the candidate report a parameter it never estimates.
    for (span, _) in slots.iter().skip(wanted.len()) {
        stale.extend(span.clone());
    }
    text.delete_lines(stale);
    for code in wanted.iter().skip(slots.len()) {
        text.append_to_block("parameters", code);
    }

    // The θ the form's features need — the power exponent, the time-varying
    // multiplier — and the `iiv_on_ruv` ω: declared when `[parameters]` lacks
    // them, left verbatim when it has them (their init and bounds are then the
    // author's, as for an existing σ above).
    for t in spec
        .exponent
        .iter()
        .chain(spec.time_varying.as_ref().map(|tv| &tv.theta))
    {
        let declared = find_decl(text, "parameters", |c| {
            THETA_RE.captures(c).is_some_and(|m| m[2] == t.name)
        })
        .is_some();
        if !declared {
            text.append_to_block("parameters", &t.render());
        }
    }
    if let Some(eta) = &spec.iiv_on_ruv {
        if omega_decl(text, &eta.name).is_none() && block_omega_decl(text, &eta.name).is_none() {
            text.append_to_block("parameters", &eta.render());
        }
    }

    // A θ or diagonal ω that only the *old* error model referenced is dead:
    // the exponent of a power form that became proportional again, the η of
    // an `iiv_on_ruv` that was removed. Checked against every line but its own
    // declaration, so a θ the rest of the model also uses stays.
    let new_idents: HashSet<String> = text
        .logical_lines("error_model")
        .iter()
        .flat_map(|(_, c)| super::IDENT_RE.find_iter(c).map(|m| m.as_str().to_string()))
        .collect();
    let mut dead = Vec::new();
    for name in old_idents.difference(&new_idents) {
        let decl = find_decl(text, "parameters", |c| {
            THETA_RE
                .captures(c)
                .or_else(|| OMEGA_RE.captures(c))
                .is_some_and(|m| m[2] == *name)
        });
        let Some((span, _)) = decl else {
            continue;
        };
        let elsewhere = text.identifiers_excluding(&span.clone().collect::<Vec<_>>());
        if !elsewhere.contains(name) {
            dead.extend(span);
        }
    }
    text.delete_lines(dead);
    Ok(())
}

/// [`ErrorSpecText::read`]: the block back in authoring form.
pub(crate) fn read_error_spec(text: &ModelText) -> Result<Option<ErrorSpecText>, String> {
    let lines = text.logical_lines("error_model");
    if lines.is_empty() && text.last_block("error_model").is_none() {
        return Ok(None);
    }
    let unreadable = |what: &str| -> String {
        format!(
            "ferx-core::edit: [error_model] {what}; only a single-endpoint `DV ~ form(σ, …)` \
             statement with bare σ names or the canonical time-varying magnitude \
             `σ * (if (TAD < c) θ else 1.0)`, plus an optional `iiv_on_ruv = ETA` line, can be \
             read back"
        )
    };
    let mut iiv: Option<String> = None;
    let mut stmt: Option<String> = None;
    for (_, code) in &lines {
        if let Some(c) = IIV_ON_RUV_RE.captures(code) {
            if iiv.replace(c[1].to_string()).is_some() {
                return Err(unreadable("has more than one `iiv_on_ruv =` line"));
            }
            continue;
        }
        if stmt.replace(code.clone()).is_some() {
            return Err(unreadable(
                "has more than one statement (a per-CMT or covariate-selected block)",
            ));
        }
    }
    let Some(stmt) = stmt else {
        return Err(unreadable("has no `DV ~ form(...)` statement"));
    };
    let caps = ERROR_STMT_RE
        .captures(&stmt)
        .ok_or_else(|| unreadable(&format!("statement `{stmt}` is not `DV ~ form(...)`")))?;
    if stmt.contains("weight") && stmt.contains('=') {
        return Err(unreadable(&format!(
            "statement `{stmt}` carries a `weight =` modifier"
        )));
    }
    let endpoint = caps[1].to_string();
    let form = ErrorForm::from_label(&caps[2])
        .ok_or_else(|| unreadable(&format!("form `{}` is not one this can read", &caps[2])))?;
    let mut args = crate::parser::model_parser::split_top_level_args(&caps[3]);

    let theta_decl = |name: &str| -> Result<ThetaDecl, String> {
        let (_, code) = find_decl(text, "parameters", |c| {
            THETA_RE.captures(c).is_some_and(|m| m[2] == *name)
        })
        .ok_or_else(|| {
            format!(
                "ferx-core::edit: [error_model] names theta `{name}`, which [parameters] does \
                 not declare"
            )
        })?;
        let m = THETA_RE.captures(&code).expect("matched above");
        let init: f64 = m[3]
            .parse()
            .map_err(|_| format!("ferx-core::edit: cannot read the init of `{code}`"))?;
        let (lower, upper) = match THETA_BOUNDS_RE.captures(&m[4]) {
            Some(b) => (
                b[1].parse::<f64>()
                    .map_err(|_| format!("ferx-core::edit: cannot read the bounds of `{code}`"))?,
                b[2].parse::<f64>()
                    .map_err(|_| format!("ferx-core::edit: cannot read the bounds of `{code}`"))?,
            ),
            None => (f64::NEG_INFINITY, f64::INFINITY),
        };
        Ok(ThetaDecl::new(name, init, lower, upper))
    };

    let exponent = if form == ErrorForm::Power {
        let p = args
            .pop()
            .ok_or_else(|| unreadable("`power(...)` has no exponent argument"))?;
        if !BARE_IDENT_RE.is_match(&p) {
            return Err(unreadable(&format!(
                "the power exponent `{p}` is an expression, not a theta name"
            )));
        }
        Some(theta_decl(&p)?)
    } else {
        None
    };
    if args.len() != form.n_sigma() {
        return Err(unreadable(&format!(
            "`{}(...)` has {} argument(s) where {} σ are expected",
            form.label(),
            args.len(),
            form.n_sigma()
        )));
    }

    let mut sigmas = Vec::with_capacity(args.len());
    let mut time_varying: Option<TimeVaryingDecl> = None;
    for (i, arg) in args.iter().enumerate() {
        let (sigma_name, tv) = if BARE_IDENT_RE.is_match(arg) {
            (arg.clone(), None)
        } else if let Some(m) = TIME_VARYING_ARG_RE.captures(arg) {
            let cutoff: f64 = m[2]
                .parse()
                .map_err(|_| unreadable(&format!("cannot read the TAD cutoff in `{arg}`")))?;
            (m[1].to_string(), Some((cutoff, m[3].to_string())))
        } else {
            return Err(unreadable(&format!(
                "argument `{arg}` is a magnitude expression this cannot read"
            )));
        };
        match (&time_varying, tv) {
            (None, Some((cutoff, theta))) => {
                if i != 0 {
                    return Err(unreadable(
                        "the time-varying magnitude is on some σ but not the first",
                    ));
                }
                time_varying = Some(TimeVaryingDecl::new(cutoff, theta_decl(&theta)?));
            }
            (Some(have), Some((cutoff, theta))) => {
                if have.cutoff != cutoff || have.theta.name != theta {
                    return Err(unreadable(
                        "the σ arguments carry different time-varying magnitudes",
                    ));
                }
            }
            (Some(_), None) => {
                return Err(unreadable(
                    "the time-varying magnitude is on some σ but not all",
                ));
            }
            (None, None) => {}
        }
        let (_, code) = find_decl(text, "parameters", |c| {
            SIGMA_RE.captures(c).is_some_and(|m| m[2] == *sigma_name)
        })
        .ok_or_else(|| {
            if find_decl(text, "parameters", |c| BLOCK_SIGMA_RE.is_match(c)).is_some() {
                unreadable(&format!("σ `{sigma_name}` is declared in a `block_sigma`"))
            } else {
                format!(
                    "ferx-core::edit: [error_model] names sigma `{sigma_name}`, which \
                     [parameters] does not declare"
                )
            }
        })?;
        let m = SIGMA_RE.captures(&code).expect("matched above");
        let init: f64 = m[3]
            .parse()
            .map_err(|_| format!("ferx-core::edit: cannot read the init of `{code}`"))?;
        sigmas.push(SigmaDecl {
            name: sigma_name,
            init,
            as_sd: SD_RE.is_match(&code),
        });
    }

    let iiv_on_ruv = match iiv {
        None => None,
        Some(eta) => {
            if block_omega_decl(text, &eta).is_some() {
                return Err(unreadable(&format!(
                    "the `iiv_on_ruv` η `{eta}` is declared in a `block_omega`"
                )));
            }
            let (_, code) = omega_decl(text, &eta).ok_or_else(|| {
                format!(
                    "ferx-core::edit: [error_model] names omega `{eta}`, which [parameters] \
                     does not declare"
                )
            })?;
            let value = omega_variance(&code)?;
            Some(EtaDecl::new(eta, value))
        }
    };

    let mut spec = ErrorSpecText::new(endpoint, form, sigmas);
    if let Some(p) = exponent {
        spec = spec.with_exponent(p);
    }
    if let Some(eta) = iiv_on_ruv {
        spec = spec.with_iiv_on_ruv(eta);
    }
    if let Some(tv) = time_varying {
        spec = spec.with_time_varying(tv);
    }
    Ok(Some(spec))
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
    let mut stale = Vec::new();
    for (span, code) in text.logical_lines("parameters") {
        if FIX_RE.is_match(&code) {
            continue;
        }
        if let Some(caps) = THETA_RE.captures(&code) {
            // A vector θ (`theta NAME[...](…)`) does not match at all — the
            // regex wants `(` straight after the name — so it is skipped here
            // for free, which is right: it has no single init to write.
            if let Some(v) = theta.get(&caps[2]).copied().filter(|v| v.is_finite()) {
                let new = format!("{}{}{}", &caps[1], num(v), &caps[4]);
                stale.extend(text.set_span(&span, &new));
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
            stale.extend(text.set_span(&span, &format!("{}{}{}", &caps[1], num(value), &caps[4])));
        } else if let Some(caps) = SIGMA_RE.captures(&code) {
            // `FitResult::sigma` is on the SD scale throughout the engine.
            let Some(sd) = sigma.get(&caps[2]).copied().filter(|v| v.is_finite()) else {
                continue;
            };
            let value = if SD_RE.is_match(&code) { sd } else { sd * sd };
            stale.extend(text.set_span(&span, &format!("{}{}{}", &caps[1], num(value), &caps[4])));
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
            stale.extend(text.set_span(
                &span,
                &format!("{}[{}]{}", &caps[1], values.join(", "), &caps[4]),
            ));
        }
    }
    text.delete_lines(stale);
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

/// The first declaration of `block` whose *logical* text satisfies `pred` —
/// the span it occupies and that text.
fn find_decl(
    text: &ModelText,
    block: &str,
    pred: impl Fn(&str) -> bool,
) -> Option<(Range<usize>, String)> {
    text.logical_lines(block)
        .into_iter()
        .find(|(_, code)| pred(code))
}

/// The names `[individual_parameters]` assigns.
fn individual_parameter_names(text: &ModelText) -> HashSet<String> {
    text.logical_lines("individual_parameters")
        .iter()
        .filter_map(|(_, code)| ASSIGN_RE.captures(code).map(|c| c[1].to_string()))
        .collect()
}

fn find_assignment(text: &ModelText, param: &str) -> Option<(Range<usize>, String)> {
    text.logical_lines("individual_parameters")
        .into_iter()
        .find_map(|(span, code)| {
            let caps = ASSIGN_RE.captures(&code)?;
            (&caps[1] == param).then(|| (span.clone(), caps[2].trim().to_string()))
        })
}

fn omega_decl(text: &ModelText, eta: &str) -> Option<(Range<usize>, String)> {
    find_decl(text, "parameters", |c| {
        OMEGA_RE.captures(c).is_some_and(|m| &m[2] == eta)
    })
}

fn block_omega_decl(text: &ModelText, eta: &str) -> Option<(Range<usize>, String)> {
    find_decl(text, "parameters", |c| {
        BLOCK_OMEGA_RE
            .captures(c)
            .is_some_and(|m| m[2].split(',').any(|n| n.trim() == eta))
    })
}

/// Every η the model declares — diagonal `omega` and `block_omega` alike.
fn declared_etas(text: &ModelText) -> Vec<String> {
    let mut out = Vec::new();
    for (_, code) in text.logical_lines("parameters") {
        if let Some(caps) = OMEGA_RE.captures(&code) {
            out.push(caps[2].to_string());
        } else if let Some(caps) = BLOCK_OMEGA_RE.captures(&code) {
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
    /// One declaration the pruner can delete: what it declares, the lines it
    /// occupies, and its logical text.
    struct Decl {
        name: String,
        span: Range<usize>,
        code: String,
    }

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

    // Declarations, by what they declare. Everything else in the file is a
    // consumer and therefore a root.
    let mut decls: Vec<Decl> = Vec::new();
    let mut relations: Vec<Decl> = Vec::new();
    // `block_omega (A, B) = […]` declares several η in one statement, and
    // declares them nowhere else — so it has to be excluded from the root scan
    // like any other declaration, or each η it names would keep *itself* alive.
    let mut blocks: Vec<(Vec<String>, Range<usize>, String)> = Vec::new();
    for (span, code) in text.logical_lines("parameters") {
        let declared = THETA_RE
            .captures(&code)
            .or_else(|| OMEGA_RE.captures(&code))
            .map(|caps| caps[2].to_string());
        if let Some(name) = declared {
            decls.push(Decl { name, span, code });
            continue;
        }
        let names = BLOCK_OMEGA_RE.captures(&code).map(|caps| {
            caps[2]
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<String>>()
        });
        if let Some(names) = names.filter(|n| !n.is_empty()) {
            blocks.push((names, span, code));
        }
    }
    for (span, code) in text.logical_lines("individual_parameters") {
        let Some(name) = ASSIGN_RE.captures(&code).map(|caps| caps[1].to_string()) else {
            continue;
        };
        decls.push(Decl { name, span, code });
    }
    for (span, code) in text.logical_lines("covariate_model") {
        let body = code.split("=>").next().unwrap_or(&code);
        let Some(name) = body.split_once('~').map(|(p, _)| p.trim().to_string()) else {
            continue;
        };
        relations.push(Decl { name, span, code });
    }

    let mut declaration_lines: Vec<usize> = Vec::new();
    for d in decls.iter().chain(relations.iter()) {
        declaration_lines.extend(d.span.clone());
    }
    for (_, span, _) in &blocks {
        declaration_lines.extend(span.clone());
    }
    let mut live = text.identifiers_excluding(&declaration_lines);

    // Grow the live set: a live name pulls in whatever its declaration reads.
    loop {
        let before = live.len();
        for d in decls.iter().chain(relations.iter()) {
            if !live.contains(&d.name) {
                continue;
            }
            for m in super::IDENT_RE.find_iter(&d.code) {
                live.insert(m.as_str().to_string());
            }
        }
        // A relation's θ is *generated* — `THETA_<PARAM>_<COV>` and its
        // `_LO`/`_HI`/`_<LEVEL>` variants never appear in the block text — so a
        // `[parameters]` declaration the desugar would defer to has no textual
        // reference to keep it alive. Protect that family, and only it: a raw
        // prefix test also matches `THETA_CL_WT2`, which belongs to a *different*
        // covariate and must still die when its own relation is dropped.
        for r in &relations {
            if !live.contains(&r.name) {
                continue;
            }
            let body = r.code.split("=>").next().unwrap_or(&r.code);
            let Some((_, rest)) = body.split_once('~') else {
                continue;
            };
            let Some(cov) = rest.trim().split_whitespace().next() else {
                continue;
            };
            let base = format!("THETA_{}_{}", r.name, cov);
            let variant = format!("{base}_");
            let generated: Vec<String> = decls
                .iter()
                .map(|d| d.name.clone())
                .filter(|n| *n == base || n.starts_with(&variant))
                .collect();
            live.extend(generated);
        }
        if live.len() == before {
            break;
        }
    }

    let dead: Vec<&Decl> = decls
        .iter()
        .chain(relations.iter())
        .filter(|d| !live.contains(&d.name))
        .collect();
    if conditional {
        if let Some(d) = dead.iter().find(|d| individual.contains(&d.span.start)) {
            return Err(format!(
                "ferx-core::edit: `{}` in [individual_parameters] is no longer referenced, but \
                 the block contains an `if`/`else`, so removing a line from it would be a guess \
                 about the branch. Edit [individual_parameters] by hand.",
                d.name
            ));
        }
    }
    let mut dead_lines: Vec<usize> = dead.iter().flat_map(|d| d.span.clone()).collect();

    // A `block_omega` that lost an η is rewritten around the ones that are
    // left, rather than kept whole (a random effect no parameter uses) or
    // deleted whole (which would take the live η with it).
    for (names, span, code) in &blocks {
        let keep: Vec<usize> = (0..names.len())
            .filter(|k| live.contains(&names[*k]))
            .collect();
        if keep.len() == names.len() {
            continue;
        }
        if keep.is_empty() {
            dead_lines.extend(span.clone());
            continue;
        }
        let decl = shrink_omega_block(code, names, &keep)?;
        dead_lines.extend(text.set_span(span, &decl));
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
