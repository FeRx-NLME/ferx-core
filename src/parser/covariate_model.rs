//! The `[covariate_model]` block: declarative covariate relationships (#1111).
//!
//! A covariate effect has always been writable directly:
//!
//! ```text
//! [individual_parameters]
//!   CL = TVCL * (WT/70)^THETA_CL_WT * exp(ETA_CL)
//! ```
//!
//! and still is. This block is *sugar* over exactly that text:
//!
//! ```text
//! [covariate_model]
//!   CL ~ WT   power(center = median)
//!   CL ~ SEX  categorical(ref = mode)
//! ```
//!
//! The point is not brevity, it is that the relations are **structured data**.
//! A covariate search (PsN `scm`, `ferx_cov_screen()`, an agent) rewrites this
//! one block and nothing else; the lines are independent, so adding or dropping
//! a relation is a pure line insert/delete rather than a regex over an
//! arbitrary expression.
//!
//! # Why `PARAM ~ COV form(...)` and not `PARAM ~ form(COV, ...)`
//!
//! The R-formula spelling — `CL ~ power(WT, center = median)`, echoing
//! `y ~ s(WT, k = 5)` — reads more naturally to anyone arriving from R, which
//! is most of this DSL's users. It was considered and rejected for two reasons.
//!
//! Once the right-hand side is a term *expression*, `CL ~ power(WT) +
//! linear(CRCL)` is what an R-literate reader will write next, because that is
//! what `~` plus function terms means everywhere else they have seen it. That
//! breaks one-relation-per-line — the property the whole block exists for — and
//! `+` reads as additive when these effects are multiplicative, so the guess
//! about the arithmetic is wrong too. A syntax that invites a construct and
//! then rejects it is worse than one that never invites it.
//!
//! And `(parameter, covariate)` is the primary key of a covariate search. This
//! spelling puts both at fixed positions at the start of the line, with the
//! form — the thing a search actually varies — as the one token that moves.
//! PsN's own config keys the same way (`[test_relations] CL=WT,SEX`, with the
//! state in a separate table), so it is also the closer translation for someone
//! coming from `scm`.
//!
//! # One θ per relation
//!
//! A relation owns the θ it declares, and two relations cannot share one — so
//! the common hand-written idiom of a *single* allometric exponent estimated
//! jointly on `CL` and `V` has no spelling here (`examples/two_cpt_oral_cov.ferx`
//! is exactly that model). This follows PsN, whose `scm` likewise gives every
//! (parameter, covariate) relation its own θ, and it is what keeps a relation a
//! self-contained line: sharing would couple two lines, and a search could no
//! longer drop one without reasoning about the other.
//!
//! The workaround is the classical block, which this one does not replace: a θ
//! declared in `[parameters]` can appear in as many `[individual_parameters]`
//! expressions as the modeller likes. A model can also mix the two — declare
//! the shared exponent classically and the rest here.
//!
//! # Where the sugar is removed
//!
//! Here, on the block *text*, before any block is read — alongside
//! `super::model_parser::apply_ode_template`'s and `apply_algebraic_structural`'s
//! source rewrites. Everything downstream (expression compilation, `Dual2`
//! sensitivities, the SAEM mu-reference detector, `covtab`, SE reporting) then
//! sees an ordinary classical model, and the generated θ are ordinary θ that
//! get estimated, bounded and reported like any other. There is no second
//! evaluation path to keep in sync.
//!
//! # Where the factor is inserted
//!
//! Not at the end of the right-hand side. A multiplicative factor commutes with
//! `exp(ETA)` numerically but **not** for the mu-reference detector SAEM uses —
//! see #619, where a typical value that is not log-linear in one θ silently
//! drops the covariate effect. So the RHS is split into top-level product
//! factors, partitioned into η/κ-bearing and not, and the covariate factors go
//! immediately before the first η/κ-bearing factor (or at the end if there is
//! none), which keeps the original factor order and puts the new factor in the
//! non-η group. When the RHS is not a top-level product at all — a sum, a logit
//! transform, a mixture branch — that is a hard error naming the explicit
//! handle, not a guess.
//!
//! # The additive operator (#1313)
//!
//! A trailing `+` makes the relation a term added to the parameter instead of
//! a factor on it — MFL's `COVARIATE(CL, WT, lin, +)`:
//!
//! ```text
//! [covariate_model]
//!   CL ~ WT linear(center = 70) +      # CL = TVCL * exp(ETA_CL) + θ*(WT - 70)
//!   CL ~ WT power(center = 70)         # the default, `*`, may also be written
//! ```
//!
//! The operator is the **last token of the line body**, before any `=>` θ
//! clause, mirroring MFL, where it is the last argument. It is not written
//! between the covariate and the form: `CL ~ WT + linear(...)` would invite
//! `CL ~ WT + AGE`, the term-expression reading this block's syntax exists to
//! avoid (see above).
//!
//! **The additive template is null at zero, and Pharmpy's is not.** Pharmpy
//! reuses the multiplicative template under `+`, so its additive linear effect
//! adds `1 + θ·(WT − median)`: a covariate at its centre adds `1` to the
//! parameter. ferx drops that leading `1` — see [`CovariateOp`] for the full
//! table and for what it costs in Pharmpy round-tripping.
//!
//! Placement is the mirror of the multiplicative rule rather than a second
//! version of it: the multiplicative factors of a parameter go into its
//! top-level product exactly as before, and the additive terms are appended
//! after it, so `CL = TVCL * <factors> * exp(ETA_CL) + <terms>`. Both kinds on
//! one parameter therefore land in one rewrite, and a multiplicative relation
//! can never be multiplied into one addend of a sum this block itself created —
//! the product is rebuilt from the original right-hand side, before any term is
//! appended. An additive relation carries no top-level-product requirement at
//! all (there is nothing to multiply into); when the right-hand side is not a
//! plain product the appended sum parenthesises it, since `if (c) a else b + t`
//! would otherwise bind the term inside the `else` arm.
//!
//! An additive term does make the typical value a sum, which the #619
//! mu-reference detector does not match — so mu-referencing switches off for
//! that parameter and SAEM falls back to the numerical M-step. That is the safe
//! direction (the effect stays in the expression; #619 was the effect being
//! *dropped*), and `model_parser` says so in a parse warning rather than
//! leaving it to be discovered in a fit.
//!
//! # Missing covariate values
//!
//! A missing covariate is `NaN` in the covariate table, and division here
//! underflows to `0.0` rather than `inf`, so an unguarded `(WT/70)^θ` on a
//! missing row would yield a silent `0` instead of a loud `NaN`. Every
//! generated effect is therefore wrapped in
//! `if (present(COV)) <effect> else <neutral>`, using the `present(...)`
//! predicate the DSL grew for exactly this. The equivalent spelling
//! `COV == COV` works too — `NaN` compares false against itself — but this text
//! is what a user reads back out of `ferx check` and diffs against a NONMEM
//! control stream, so it says what it means.
//!
//! The neutral element is **per operator** — `1.0` for a factor, `0.0` for an
//! added term ([`CovariateOp::neutral`]). Sharing one would give a subject with
//! a missing covariate a silent `+1` on the parameter.

use std::collections::{HashMap, HashSet};

use crate::types::{
    CovariateDecl, CovariateForm, CovariateKind, CovariateLevels, CovariateModelSpec, CovariateOp,
    CovariateRelation, CovariateStat, CovariateSummary, CovariateTheta,
};

/// Data-derived covariate statistics threaded into a second parse, keyed by
/// covariate name. Empty on the first parse, filled by
/// [`crate::api::bind_covariate_stats`].
pub type CovariateStatBindings = HashMap<String, CovariateSummary>;

/// Everything the desugar needs to know about the rest of the model.
pub(crate) struct CovariateModelContext<'a> {
    /// `[covariates]` declarations. The block is **authoritative** here: unlike
    /// the lenient classical path, which warns about an undeclared covariate
    /// and reads it anyway, a relation on an undeclared covariate is an error —
    /// the form (`categorical` vs continuous) and the levels are read off this
    /// declaration, so guessing would generate a different θ vector per dataset.
    pub decls: &'a [CovariateDecl],
    /// η and κ names, so a factor can be classified as random-effect-bearing.
    pub eta_kappa_names: &'a HashSet<String>,
    /// θ already declared in `[parameters]`. An auto-generated name that
    /// collides with one of these defers to it (the modeller has stated the
    /// init and bounds they want); an explicit `=> NAME(...)` that collides is
    /// an error, since it would declare the same θ twice.
    pub declared_thetas: &'a HashSet<String>,
    /// Resolved statistics, empty until the model is bound to a population.
    pub stats: &'a CovariateStatBindings,
}

/// Rewrite `[parameters]` and `[individual_parameters]` for the relations in
/// `block_lines`, returning the structured echo.
///
/// Relations that still need data-derived statistics are parsed, validated and
/// recorded, but **not** desugared — they carry no θ and change no expression.
/// Every entry point rejects a model that still carries one, so a symbolic
/// `center = median` can never silently fit as if the covariate were absent.
pub(crate) fn apply_covariate_model(
    block_lines: &[String],
    parameters: &mut Vec<String>,
    individual_parameters: &mut [String],
    ctx: &CovariateModelContext<'_>,
) -> Result<CovariateModelSpec, String> {
    let indiv_names = top_level_assignments(individual_parameters);
    let relations = parse_relations(block_lines, ctx, &indiv_names)?;

    let mut generated_thetas = Vec::new();
    for rel in &relations {
        for theta in &rel.thetas {
            // An auto-generated name the modeller already declared defers to
            // their declaration — re-emitting it would be a duplicate θ.
            if ctx.declared_thetas.contains(&theta.name) {
                continue;
            }
            generated_thetas.push(theta_declaration(theta));
        }
    }
    parameters.extend(generated_thetas.iter().cloned());

    // Group the effects by parameter so one rewrite pass handles every relation
    // on a given parameter, in declaration order, keeping the multiplicative
    // and additive ones apart: the product is rebuilt from the original
    // right-hand side and the terms are appended after it, so a factor can
    // never end up inside one addend of a sum this block created.
    let mut effects: Vec<ParameterEffects> = Vec::new();
    for rel in &relations {
        let Some(text) = relation_effect(rel) else {
            continue;
        };
        let entry = match effects.iter_mut().find(|e| e.parameter == rel.parameter) {
            Some(e) => e,
            None => {
                effects.push(ParameterEffects {
                    parameter: rel.parameter.clone(),
                    factors: Vec::new(),
                    terms: Vec::new(),
                });
                effects.last_mut().expect("just pushed")
            }
        };
        match rel.op {
            CovariateOp::Multiply => entry.factors.push(text),
            CovariateOp::Add => entry.terms.push(text),
        }
    }
    // Classify factors against the η/κ names *plus* every intermediate that
    // reads one, so a factor is not mistaken for η-free (see below).
    let random_bearing = random_bearing_names(individual_parameters, ctx.eta_kappa_names);
    for e in &effects {
        insert_effects(
            individual_parameters,
            &e.parameter,
            &e.factors,
            &e.terms,
            &random_bearing,
        )?;
    }

    Ok(CovariateModelSpec {
        desugared_individual_parameters: individual_parameters
            .iter()
            .map(|l| l.trim_end().to_string())
            .collect(),
        generated_thetas,
        relations,
    })
}

/// The generated text of every relation on one `[individual_parameters]`
/// name, split by operator — the multiplicative factors in declaration order
/// and the additive terms in declaration order.
struct ParameterEffects {
    parameter: String,
    factors: Vec<String>,
    terms: Vec<String>,
}

// ── Relation parsing ───────────────────────────────────────────────────────

/// Parse every line of the block into a [`CovariateRelation`], resolving the
/// centering statistic and default θ bounds where the data allows.
fn parse_relations(
    lines: &[String],
    ctx: &CovariateModelContext<'_>,
    indiv_names: &[String],
) -> Result<Vec<CovariateRelation>, String> {
    let mut out: Vec<CovariateRelation> = Vec::new();
    let mut seen_theta: HashSet<String> = HashSet::new();
    for line in lines {
        let line = strip_comment(line);
        if line.is_empty() {
            continue;
        }
        let rel = parse_relation_line(line, ctx, indiv_names)?;
        if out
            .iter()
            .any(|r| r.parameter == rel.parameter && r.covariate == rel.covariate)
        {
            return Err(format!(
                "[covariate_model]: relation `{} ~ {}` is declared more than once — one line \
                 per (parameter, covariate) pair",
                rel.parameter, rel.covariate
            ));
        }
        for theta in &rel.thetas {
            if !seen_theta.insert(theta.name.clone()) {
                return Err(format!(
                    "[covariate_model]: two relations both generate the θ `{}`. Each relation \
                     owns its own θ — the block cannot share one across relations (a shared \
                     allometric exponent on CL and V, say). Give them distinct names with \
                     `=> NAME(init, lower, upper)`, or write the shared-θ factor classically \
                     in [individual_parameters], where one θ can appear in as many \
                     expressions as you like.",
                    theta.name
                ));
            }
        }
        out.push(rel);
    }
    Ok(out)
}

/// `PARAM ~ COV form(kwargs) [*|+] [=> THETA(init, lower, upper)[, ...]]`
fn parse_relation_line(
    line: &str,
    ctx: &CovariateModelContext<'_>,
    indiv_names: &[String],
) -> Result<CovariateRelation, String> {
    let (body, theta_clause) = match line.split_once("=>") {
        Some((b, t)) => (b.trim(), Some(t.trim())),
        None => (line, None),
    };
    let (body, op) = split_trailing_op(body, line)?;
    let Some((param, rest)) = body.split_once('~') else {
        return Err(format!(
            "[covariate_model]: expected `PARAM ~ COV form(...)`, got `{line}`"
        ));
    };
    let param = param.trim();
    if param.is_empty() {
        return Err(format!(
            "[covariate_model]: missing parameter name in `{line}`"
        ));
    }
    if !indiv_names.iter().any(|n| n == param) {
        return Err(format!(
            "[covariate_model]: `{param}` is not a top-level [individual_parameters] name \
             (declared: {})",
            join_names(indiv_names)
        ));
    }
    let rest = rest.trim();
    let (covariate, form_src) = match rest.find(char::is_whitespace) {
        Some(i) => (rest[..i].trim(), rest[i..].trim()),
        None => (rest, ""),
    };
    if form_src.is_empty() {
        return Err(format!(
            "[covariate_model]: `{param} ~ {covariate}` states no form — expected one of {}",
            form_list()
        ));
    }
    let decl = ctx
        .decls
        .iter()
        .find(|d| d.name == covariate)
        .ok_or_else(|| {
            format!(
                "[covariate_model]: covariate `{covariate}` is not declared in [covariates] \
                 (declared: {}). The block is authoritative: a relation's form and levels are \
                 read off the declaration, so it must be stated.",
                join_names(&ctx.decls.iter().map(|d| d.name.clone()).collect::<Vec<_>>())
            )
        })?;

    let (form, kwargs) = parse_form(form_src)?;
    check_kind(&form, decl)?;

    let center = center_argument(&form, &kwargs, param, covariate)?;
    let fix = match kwargs.get("fix") {
        Some(v) => Some(v.literal().ok_or_else(|| {
            format!(
                "[covariate_model]: `fix` takes a number, got `{}`",
                v.label()
            )
        })?),
        None => None,
    };
    let summary = ctx.stats.get(covariate);
    let resolved_center = match center {
        Some(CovariateStat::Literal(v)) => Some(v),
        Some(stat) => summary.map(|s| s.value_of(stat)),
        None => None,
    };
    if let Some(c) = resolved_center {
        check_center_domain(&form, c, param, covariate)?;
    }

    let mut rel = CovariateRelation {
        parameter: param.to_string(),
        covariate: covariate.to_string(),
        form,
        op,
        center,
        resolved_center,
        fix,
        thetas: Vec::new(),
        source_line: line.to_string(),
    };

    let explicit = match theta_clause {
        Some(clause) => Some(parse_theta_clause(clause, ctx)?),
        None => None,
    };
    rel.thetas = build_thetas(&rel, decl, summary, explicit.as_deref())?;
    Ok(rel)
}

/// Split the trailing `*` / `+` operator token off the line body (#1313).
///
/// The operator is the **last** token of the body, after the form and before
/// any `=>` clause (which the caller has already removed) — the position MFL
/// puts it in, `COVARIATE(CL, WT, lin, +)`. Nothing else a relation line can
/// end with is a bare `+` or `*`: every form ends in an identifier or a `)`,
/// and `expr("WT*2")` keeps its operators inside the quotes.
///
/// `*` is the default and may be written explicitly, so a generated file can
/// state the combination on every line rather than only on the additive ones.
fn split_trailing_op<'a>(body: &'a str, line: &str) -> Result<(&'a str, CovariateOp), String> {
    let body = body.trim();
    let (rest, op) = match body.chars().next_back() {
        Some('+') => (&body[..body.len() - 1], CovariateOp::Add),
        Some('*') => (&body[..body.len() - 1], CovariateOp::Multiply),
        _ => return Ok((body, CovariateOp::Multiply)),
    };
    let rest = rest.trim();
    if rest.is_empty() || rest.ends_with('~') {
        return Err(format!(
            "[covariate_model]: `{line}` states the `{}` operator but no form — expected \
             `PARAM ~ COV form(...) {}`",
            op.label(),
            op.label()
        ));
    }
    Ok((rest, op))
}

/// Reject a centring constant the relation's own algebra cannot use.
///
/// `power` and `linear_relative` divide by the centre, and division by a zero
/// (or missing) value **underflows to `0.0`** in this engine rather than
/// blowing up — a `center = 0` would silently turn the whole covariate factor
/// into `0` or `1`. A negative centre in `power` sends a positive covariate
/// through `(negative)^θ`, which is `NaN` for a non-integer θ. Both are caught
/// here, at parse time, rather than at the first evaluation.
fn check_center_domain(
    form: &CovariateForm,
    center: f64,
    param: &str,
    covariate: &str,
) -> Result<(), String> {
    let bad = |why: &str| {
        Err(format!(
            "[covariate_model]: `{param} ~ {covariate} {}` resolves its centre to {center}, {why}",
            form.label()
        ))
    };
    if !center.is_finite() {
        return bad("which is not a finite number. State it as a literal (`center = 70`).");
    }
    match form {
        CovariateForm::Power if center <= 0.0 => bad(
            "but `power` evaluates `(COV / centre)^θ` — a zero centre divides to `0` (this \
             engine underflows rather than raising) and a negative one makes the base negative, \
             which is `NaN` for a non-integer θ. Use a positive centre.",
        ),
        CovariateForm::LinearRelative if center == 0.0 => bad(
            "but `linear_relative` evaluates `1 + θ·(COV / centre − 1)` — dividing by zero \
             underflows to `0` here, so the relation would silently contribute `1 − θ` for \
             every subject. Use a non-zero centre, or `linear`, which subtracts instead.",
        ),
        _ => Ok(()),
    }
}

/// `power(center = median)` → the form and its keyword arguments.
fn parse_form(src: &str) -> Result<(CovariateForm, HashMap<String, CovariateStat>), String> {
    let src = src.trim();
    let (head, body) = match src.find('(') {
        Some(i) => {
            let body = src[i..]
                .strip_prefix('(')
                .and_then(|r| r.strip_suffix(')'))
                .ok_or_else(|| {
                    format!("[covariate_model]: unbalanced parentheses in form `{src}`")
                })?;
            (src[..i].trim(), body.trim())
        }
        None => (src, ""),
    };
    let form = match head.to_lowercase().as_str() {
        "none" => CovariateForm::None,
        "linear" => CovariateForm::Linear,
        "linear_relative" => CovariateForm::LinearRelative,
        "exponential" => CovariateForm::Exponential,
        "power" => CovariateForm::Power,
        "hockey" => CovariateForm::Hockey,
        "categorical" => CovariateForm::Categorical,
        "categorical2" => CovariateForm::Categorical2,
        "expr" => {
            let text = body
                .trim()
                .strip_prefix('"')
                .and_then(|r| r.strip_suffix('"'))
                .ok_or_else(|| {
                    "[covariate_model]: `expr` takes a quoted expression, e.g. \
                     `expr(\"(WT/70)^0.75\")`"
                        .to_string()
                })?;
            if text.trim().is_empty() {
                return Err("[covariate_model]: `expr(\"\")` states no expression".to_string());
            }
            return Ok((CovariateForm::Expr(text.to_string()), HashMap::new()));
        }
        other => {
            return Err(format!(
                "[covariate_model]: unknown form `{other}`{} — expected one of {}",
                did_you_mean(other),
                form_list()
            ))
        }
    };
    Ok((form, parse_kwargs(body, head)?))
}

/// `center = median, fix = 0.75` → a name→value map. Values are a number or one
/// of the data-derived statistics.
fn parse_kwargs(body: &str, form: &str) -> Result<HashMap<String, CovariateStat>, String> {
    let mut out = HashMap::new();
    for part in body.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let Some((key, value)) = part.split_once('=') else {
            return Err(format!(
                "[covariate_model]: `{form}(...)` takes `key = value` arguments, got `{part}`"
            ));
        };
        let key = key.trim().to_lowercase();
        if !matches!(key.as_str(), "center" | "breakpoint" | "ref" | "fix") {
            return Err(format!(
                "[covariate_model]: unknown argument `{key}` to `{form}(...)` — expected \
                 center, breakpoint, ref or fix"
            ));
        }
        let value = value.trim();
        let stat = match value.to_lowercase().as_str() {
            "median" => CovariateStat::Median,
            "mean" => CovariateStat::Mean,
            "min" | "minimum" => CovariateStat::Min,
            "max" | "maximum" => CovariateStat::Max,
            "mode" => CovariateStat::Mode,
            _ => CovariateStat::Literal(value.parse::<f64>().map_err(|_| {
                format!(
                    "[covariate_model]: `{key} = {value}` is neither a number nor one of \
                     median, mean, min, max, mode"
                )
            })?),
        };
        if out.insert(key.clone(), stat).is_some() {
            return Err(format!(
                "[covariate_model]: `{key}` is given more than once in `{form}(...)`"
            ));
        }
    }
    Ok(out)
}

/// The centering constant a form takes, under the keyword that form spells it
/// with: `center` for the continuous forms, `breakpoint` for `hockey`, `ref`
/// for `categorical`.
fn center_argument(
    form: &CovariateForm,
    kwargs: &HashMap<String, CovariateStat>,
    param: &str,
    covariate: &str,
) -> Result<Option<CovariateStat>, String> {
    let (keyword, default) = match form {
        CovariateForm::None | CovariateForm::Expr(_) => {
            // `fix` joins the centring keys here: both forms declare no θ, so
            // `none(fix = 0.5)` would parse, set `rel.fix`, and then have
            // `apply_fix` run over an empty θ list — no effect, no diagnostic.
            for key in ["center", "breakpoint", "ref", "fix"] {
                if kwargs.contains_key(key) {
                    return Err(format!(
                        "[covariate_model]: `{}` takes no `{key}`",
                        form.label()
                    ));
                }
            }
            return Ok(None);
        }
        CovariateForm::Hockey => ("breakpoint", CovariateStat::Median),
        CovariateForm::Categorical | CovariateForm::Categorical2 => ("ref", CovariateStat::Mode),
        _ => ("center", CovariateStat::Median),
    };
    for key in ["center", "breakpoint", "ref"] {
        if key != keyword && kwargs.contains_key(key) {
            return Err(format!(
                "[covariate_model]: `{}` centres on `{keyword}`, not `{key}` \
                 (in `{param} ~ {covariate}`)",
                form.label()
            ));
        }
    }
    Ok(Some(kwargs.get(keyword).copied().unwrap_or(default)))
}

/// A `categorical` form must sit on a categorical covariate and every other
/// form on a continuous one. Silently allowing the mismatch would generate a
/// factor keyed on a level that does not exist, or a power of a 0/1 code.
fn check_kind(form: &CovariateForm, decl: &CovariateDecl) -> Result<(), String> {
    if matches!(form, CovariateForm::None | CovariateForm::Expr(_)) {
        return Ok(());
    }
    match (form.is_categorical(), decl.kind) {
        (true, CovariateKind::Continuous) => Err(format!(
            "[covariate_model]: `{}(...)` on `{}`, which [covariates] declares continuous",
            form.label(),
            decl.name
        )),
        (false, CovariateKind::Categorical) => Err(format!(
            "[covariate_model]: `{}` is a continuous form, but [covariates] declares `{}` \
             categorical — use `categorical(ref = ...)`",
            form.label(),
            decl.name
        )),
        _ => Ok(()),
    }
}

/// `=> NAME(init, lower, upper)[, NAME(...)]` — an explicit θ declaration,
/// overriding the form's defaults. One entry per θ the form generates.
fn parse_theta_clause(
    clause: &str,
    ctx: &CovariateModelContext<'_>,
) -> Result<Vec<CovariateTheta>, String> {
    let mut out = Vec::new();
    for part in split_top_level(clause, ',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (name, args) = part
            .split_once('(')
            .and_then(|(n, a)| a.strip_suffix(')').map(|a| (n.trim(), a)))
            .ok_or_else(|| {
                format!("[covariate_model]: expected `=> NAME(init, lower, upper)`, got `{part}`")
            })?;
        let nums: Vec<f64> = args
            .split(',')
            .map(|t| {
                t.trim().parse::<f64>().map_err(|_| {
                    format!(
                        "[covariate_model]: `{}` is not a number in `{part}`",
                        t.trim()
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        let [init, lower, upper] = nums[..] else {
            return Err(format!(
                "[covariate_model]: `{name}(...)` needs exactly (init, lower, upper), got \
                 {} values",
                nums.len()
            ));
        };
        if ctx.declared_thetas.contains(name) {
            return Err(format!(
                "[covariate_model]: θ `{name}` is already declared in [parameters] — drop the \
                 `=> {name}(...)` clause to use that declaration, or rename one of them"
            ));
        }
        out.push(CovariateTheta {
            name: name.to_string(),
            init,
            lower,
            upper,
            fixed: false,
            level: None,
        });
    }
    if out.is_empty() {
        return Err("[covariate_model]: empty `=>` clause".to_string());
    }
    Ok(out)
}

// ── θ construction (PsN defaults) ──────────────────────────────────────────

/// Build the θ a relation declares.
///
/// Returns an empty vector — leaving the relation *unresolved* — when the form
/// needs data-derived statistics that are not bound yet. `none` and `expr(...)`
/// declare no θ and are never unresolved.
fn build_thetas(
    rel: &CovariateRelation,
    decl: &CovariateDecl,
    summary: Option<&CovariateSummary>,
    explicit: Option<&[CovariateTheta]>,
) -> Result<Vec<CovariateTheta>, String> {
    // `levels = auto` is the opt-in to data-derived levels: with no population
    // bound yet the θ count is simply not known, which is an unresolved
    // relation, not an error.
    if rel.form.is_categorical()
        && matches!(decl.levels, Some(CovariateLevels::Auto))
        && summary.is_none()
    {
        return Ok(Vec::new());
    }
    let expected = expected_theta_count(rel, decl, summary)?;
    if expected == 0 && rel.form.is_categorical() {
        // Levels are known (declared, or discovered for `levels = auto`) and
        // there is exactly one, so there is no contrast to estimate. Without
        // this the relation would carry no θ, `needs_data()` would read that as
        // *unresolved*, and the user would be told to supply `--data` — which
        // can never help.
        let levels = levels_of(decl, summary).unwrap_or_default();
        return Err(format!(
            "[covariate_model]: `{} ~ {} {}(...)` has nothing to estimate — `{}` has \
             {} level{} ({}), and a categorical relation spends one θ per *non-reference* level. \
             Declare the levels the data actually carries (`{} categorical(levels = [...])` in \
             [covariates]), or drop the relation.",
            rel.parameter,
            rel.covariate,
            rel.form.label(),
            decl.name,
            levels.len(),
            if levels.len() == 1 { "" } else { "s" },
            join_levels(&levels),
            decl.name,
        ));
    }
    if expected == 0 {
        if let Some(explicit) = explicit {
            return Err(format!(
                "[covariate_model]: `{}` declares no θ, so the `=> {}(...)` clause has nothing \
                 to name",
                rel.form.label(),
                explicit[0].name
            ));
        }
        return Ok(Vec::new());
    }
    if let Some(explicit) = explicit {
        if explicit.len() != expected {
            return Err(format!(
                "[covariate_model]: `{} ~ {} {}` generates {expected} θ but the `=>` clause \
                 names {}",
                rel.parameter,
                rel.covariate,
                rel.form.label(),
                explicit.len()
            ));
        }
        let mut out = explicit.to_vec();
        apply_fix(&mut out, rel);
        if rel.form.is_categorical() {
            for (theta, level) in out
                .iter_mut()
                .zip(non_reference_levels(rel, decl, summary)?)
            {
                theta.level = Some(level);
            }
        }
        return Ok(out);
    }

    // Every default below needs the centering constant; and the two linear
    // families need the covariate's spread, which only the data has.
    let Some(center) = rel.resolved_center else {
        return Ok(Vec::new());
    };
    let mut out = match rel.form {
        CovariateForm::Categorical | CovariateForm::Categorical2 => {
            let base = format!("THETA_{}_{}", rel.parameter, rel.covariate);
            // The two shapes differ by `θ_cat2 = 1 + θ_cat`, so `categorical2`
            // takes the *image* of `categorical`'s defaults under exactly that
            // map: bounds (−1, 5) → (0, 6) — which is what Pharmpy's `cat2`
            // uses verbatim — and init −0.001 → 0.999, so the two forms start
            // at the identical model. (Pharmpy's own `cat2` init is 1.01; ferx
            // already differs on the `cat` init's sign, following PsN.)
            let (init, lower, upper) = match rel.form {
                CovariateForm::Categorical2 => (0.999, 0.0, 6.0),
                // PsN's categorical θ is null at 0 and bounded symmetrically
                // about it, so the parameterization has no preferred sign.
                _ => (-0.001, -1.0, 5.0),
            };
            non_reference_levels(rel, decl, summary)?
                .into_iter()
                .map(|level| CovariateTheta {
                    name: format!("{base}_{}", level_suffix(level)),
                    init,
                    lower,
                    upper,
                    fixed: false,
                    level: Some(level),
                })
                .collect()
        }
        CovariateForm::Exponential | CovariateForm::Power => vec![CovariateTheta {
            name: format!("THETA_{}_{}", rel.parameter, rel.covariate),
            init: 0.001,
            lower: -100.0,
            upper: 1e6,
            fixed: false,
            level: None,
        }],
        CovariateForm::Linear | CovariateForm::LinearRelative | CovariateForm::Hockey => {
            let Some(summary) = summary else {
                return Ok(Vec::new());
            };
            linear_family_thetas(rel, summary, center)?
        }
        CovariateForm::None | CovariateForm::Expr(_) => Vec::new(),
    };
    apply_fix(&mut out, rel);
    Ok(out)
}

/// PsN's default init/bounds for the linear family.
///
/// The bounds are not cosmetic: `θ ∈ [1/(med−max), 1/(med−min)]` is exactly
/// what keeps `1 + θ·(COV − med)` positive over the observed covariate range,
/// which is why they are adopted verbatim rather than re-invented. A constant
/// covariate makes both bounds infinite, so it is rejected here instead of
/// producing a θ the optimiser cannot move.
fn linear_family_thetas(
    rel: &CovariateRelation,
    summary: &CovariateSummary,
    center: f64,
) -> Result<Vec<CovariateTheta>, String> {
    let below = center - summary.min;
    let above = center - summary.max;
    // The default bounds only order themselves when the centre sits *strictly
    // inside* the observed range: `below > 0 > above` puts `1/above` below
    // `1/below`. A constant covariate makes one of them infinite; a centre
    // outside the range makes both the same sign, which emits `lower > upper`
    // (e.g. centre 100 over a 50..90 range → `(0.1, 0.02)`).
    if !(below > 0.0 && above < 0.0) {
        return Err(format!(
            "[covariate_model]: `{} ~ {} {}` needs its centre ({center}) to lie strictly inside \
             the observed range of `{}`, which spans [{}, {}] in the data — the PsN default \
             bounds 1/(centre − min) and 1/(centre − max) are only ordered (and only keep the \
             covariate factor positive) there. State the θ explicitly with \
             `=> NAME(init, lower, upper)`, or drop the relation.",
            rel.parameter,
            rel.covariate,
            rel.form.label(),
            rel.covariate,
            summary.min,
            summary.max
        ));
    }
    // `linear_relative` is `linear` reparameterized as θ_rel = centre · θ_abs,
    // so its init and bounds are the linear ones scaled by the centre — the two
    // forms then start the optimiser at the same place and admit the same
    // covariate factors, and differ only in the scale θ reports on.
    let scale = if matches!(rel.form, CovariateForm::LinearRelative) {
        center
    } else {
        1.0
    };
    let base = format!("THETA_{}_{}", rel.parameter, rel.covariate);
    Ok(match rel.form {
        CovariateForm::Hockey => vec![
            CovariateTheta {
                name: format!("{base}_LO"),
                init: 0.001 / below,
                lower: -1e6,
                upper: 1.0 / below,
                fixed: false,
                level: None,
            },
            CovariateTheta {
                name: format!("{base}_HI"),
                init: 0.001 / above,
                lower: 1.0 / above,
                upper: 1e6,
                fixed: false,
                level: None,
            },
        ],
        _ => {
            // Scaling by a negative centre (`linear_relative` on a covariate
            // that straddles zero) reverses the interval, so re-order rather
            // than emit `lower > upper`.
            let (lower, upper) = ordered(scale * 1.0 / above, scale / below);
            vec![CovariateTheta {
                name: base,
                init: scale * 0.001 / below,
                lower,
                upper,
                fixed: false,
                level: None,
            }]
        }
    })
}

/// `(min, max)` of two bounds — a negative `linear_relative` centre flips them.
fn ordered(a: f64, b: f64) -> (f64, f64) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// How many θ a relation generates, or an error when the answer is unknowable
/// from the file alone (a `categorical` with neither declared nor bound levels).
fn expected_theta_count(
    rel: &CovariateRelation,
    decl: &CovariateDecl,
    summary: Option<&CovariateSummary>,
) -> Result<usize, String> {
    Ok(match rel.form {
        CovariateForm::None | CovariateForm::Expr(_) => 0,
        CovariateForm::Hockey => 2,
        CovariateForm::Categorical | CovariateForm::Categorical2 => {
            match levels_of(decl, summary) {
                Some(levels) => levels.len().saturating_sub(1),
                None => {
                    return Err(format!(
                        "[covariate_model]: `{}(...)` on `{}` needs its levels, but [covariates] \
                         declares none. Write `{} categorical(levels = [0, 1])` for a θ vector \
                         fixed by the file, or `levels = auto` to read them off the data.",
                        rel.form.label(),
                        decl.name,
                        decl.name
                    ))
                }
            }
        }
        _ => 1,
    })
}

/// The covariate's levels: as declared, or as discovered for `levels = auto`.
/// `None` when `auto` is asked for and no data is bound yet.
fn levels_of(decl: &CovariateDecl, summary: Option<&CovariateSummary>) -> Option<Vec<f64>> {
    match decl.levels.as_ref()? {
        CovariateLevels::Declared(v) => Some(v.clone()),
        CovariateLevels::Auto => summary.map(|s| s.levels.clone()),
    }
}

/// Every level but the reference one, in declaration order — one θ each.
fn non_reference_levels(
    rel: &CovariateRelation,
    decl: &CovariateDecl,
    summary: Option<&CovariateSummary>,
) -> Result<Vec<f64>, String> {
    let levels = levels_of(decl, summary).unwrap_or_default();
    let Some(reference) = rel.resolved_center else {
        return Ok(Vec::new());
    };
    if !levels.contains(&reference) {
        return Err(format!(
            "[covariate_model]: reference level {reference} of `{} ~ {}` is not one of the \
             declared levels {levels:?}",
            rel.parameter, rel.covariate
        ));
    }
    Ok(levels.into_iter().filter(|l| *l != reference).collect())
}

/// `fix = v` pins every θ of the relation at `v`, declared `FIX`.
fn apply_fix(thetas: &mut [CovariateTheta], rel: &CovariateRelation) {
    let Some(v) = rel.fix else { return };
    for theta in thetas {
        theta.init = v;
        theta.fixed = true;
    }
}

/// `THETA_CL_SEX_1` / `THETA_CL_SEX_M1` / `THETA_CL_SEX_1_5` — a level rendered
/// as a name fragment, with `-` and `.` spelled out so the result is a valid
/// identifier.
fn level_suffix(level: f64) -> String {
    let text = if level.fract() == 0.0 && level.abs() < 1e15 {
        format!("{}", level as i64)
    } else {
        format!("{level}")
    };
    text.replace('-', "M").replace('.', "_")
}

/// The `[parameters]` line for a generated θ.
fn theta_declaration(theta: &CovariateTheta) -> String {
    format!(
        "  theta {}({}, {}, {}){}",
        theta.name,
        fmt(theta.init),
        fmt(theta.lower),
        fmt(theta.upper),
        if theta.fixed { " FIX" } else { "" }
    )
}

// ── Factor generation ──────────────────────────────────────────────────────

/// The effect text a relation contributes — a multiplicative factor under `*`,
/// an added term under `+` — already guarded against a missing covariate
/// value. `None` for `none` and for a relation still waiting on data.
///
/// The two operators share the *shape* of each form and differ in its null:
/// the multiplicative template is `1 + …` (or `exp(…)` / `(…)^θ`), whose
/// neutral value is `1`, and the additive one is the same expression minus that
/// `1`, so a covariate at its centre contributes `0` and so does a θ at the
/// *form's* null — `0` everywhere except `categorical2`, whose θ is the factor
/// itself and whose null is `1` (#1312), hence its `θ_k - 1`. This is
/// where ferx parts company with Pharmpy, which reuses the multiplicative
/// template verbatim under `+` — see [`CovariateOp`].
fn relation_effect(rel: &CovariateRelation) -> Option<String> {
    let cov = &rel.covariate;
    let add = rel.op == CovariateOp::Add;
    let inner = match &rel.form {
        CovariateForm::None => return None,
        // `expr(...)` is verbatim by definition: the operator says how the
        // modeller's own expression combines, and does not rewrite it.
        CovariateForm::Expr(text) => format!("({text})"),
        _ if rel.needs_data() => return None,
        CovariateForm::Linear => {
            let c = fmt(rel.resolved_center?);
            let theta = &rel.thetas[0].name;
            if add {
                format!("({theta} * ({cov} - {c}))")
            } else {
                format!("(1 + {theta} * ({cov} - {c}))")
            }
        }
        CovariateForm::LinearRelative => {
            let c = fmt(rel.resolved_center?);
            let theta = &rel.thetas[0].name;
            if add {
                format!("({theta} * ({cov} / {c} - 1))")
            } else {
                format!("(1 + {theta} * ({cov} / {c} - 1))")
            }
        }
        CovariateForm::Exponential => {
            let c = fmt(rel.resolved_center?);
            let theta = &rel.thetas[0].name;
            if add {
                format!("(exp({theta} * ({cov} - {c})) - 1)")
            } else {
                format!("exp({theta} * ({cov} - {c}))")
            }
        }
        CovariateForm::Power => {
            let c = fmt(rel.resolved_center?);
            let theta = &rel.thetas[0].name;
            if add {
                format!("(({cov} / {c})^{theta} - 1)")
            } else {
                format!("({cov} / {c})^{theta}")
            }
        }
        CovariateForm::Hockey => {
            let b = fmt(rel.resolved_center?);
            let (lo, hi) = (&rel.thetas[0].name, &rel.thetas[1].name);
            if add {
                format!("(if ({cov} <= {b}) {lo} * ({cov} - {b}) else {hi} * ({cov} - {b}))")
            } else {
                format!(
                    "(if ({cov} <= {b}) 1 + {lo} * ({cov} - {b}) else 1 + {hi} * ({cov} - {b}))"
                )
            }
        }
        CovariateForm::Categorical | CovariateForm::Categorical2 => {
            // `categorical` reads its θ as an offset from the reference
            // (`1 + θ_k`), `categorical2` as the factor itself (`θ_k`). Only
            // the non-reference branch differs.
            let offset = matches!(rel.form, CovariateForm::Categorical);
            let mut expr = String::from("(");
            for theta in &rel.thetas {
                // Null-subtraction under `+` is the same one every other
                // additive template applies, and it is per *form*, not shared:
                // `categorical`'s multiplicative branch is `1 + θ_k`, so its
                // additive one is `θ_k`; `categorical2`'s is `θ_k` against a
                // reference of `1`, so its additive one is `θ_k - 1`. That
                // keeps `θ_cat2 = 1 + θ_cat` an exact reparameterization under
                // `+` too — and keeps `cat +` and `cat2 +` from being the same
                // text, which would make them duplicate search candidates.
                let body = match (add, offset) {
                    (false, true) => format!("1 + {}", theta.name),
                    (false, false) | (true, true) => theta.name.clone(),
                    (true, false) => format!("{} - 1", theta.name),
                };
                expr.push_str(&format!("if ({cov} == {}) {body} else ", fmt(theta.level?)));
            }
            // The reference level, and anything not listed, is the null.
            expr.push_str(if add { "0)" } else { "1)" });
            expr
        }
    };
    // A missing covariate takes the neutral branch — without this the relation
    // would silently contribute `0` (division by a missing value underflows to
    // zero here, it does not blow up). The neutral element is the operator's,
    // not a shared `1.0`: a missing covariate on an additive relation must
    // contribute nothing, and `+ 1.0` is not nothing.
    Some(format!(
        "(if (present({cov})) {inner} else {:.1})",
        rel.op.neutral()
    ))
}

/// `eta_kappa`, plus every top-level `[individual_parameters]` variable whose
/// definition transitively reads one.
///
/// [`insert_factors`] classifies a factor by the *names* it mentions, so an η
/// reached through an intermediate —
///
/// ```text
/// CLI = exp(ETA_CL)
/// CL  = TVCL * CLI
/// ```
///
/// — would otherwise look η-free, and the covariate factor would be appended
/// *after* `CLI`: the placement #619 is about, silently rather than as the
/// `COV_<PARAM>` hard error. Adding `CLI` to the set puts the factor back in the
/// non-η group.
///
/// Iterated to a fixed point rather than taken in source order, since an
/// `if { ... }` block can reassign a name after its first definition.
fn random_bearing_names(lines: &[String], eta_kappa: &HashSet<String>) -> HashSet<String> {
    let mut set = eta_kappa.clone();
    loop {
        let mut grew = false;
        for line in lines {
            let text = strip_comment(line);
            let Some(name) = assignment_target(text) else {
                continue;
            };
            if set.contains(&name) {
                continue;
            }
            let rhs = text.split_once('=').map(|(_, r)| r).unwrap_or("");
            if mentions_any(rhs, &set) {
                set.insert(name);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    set
}

/// Rewrite `param`'s right-hand side with the relations on it: `factors`
/// multiplied in immediately before the first η/κ-bearing top-level factor,
/// then `terms` appended as a sum.
///
/// The order is the point. The product is rebuilt from the **original**
/// right-hand side, so the top-level-product requirement is asked of the model
/// the user wrote, not of the sum this function is about to create — and a
/// multiplicative relation can never be multiplied into one addend of an
/// additive relation declared before it, whatever order the block lines are in.
fn insert_effects(
    lines: &mut [String],
    param: &str,
    factors: &[String],
    terms: &[String],
    eta_kappa: &HashSet<String>,
) -> Result<(), String> {
    let idx = assignment_line(lines, param).ok_or_else(|| {
        format!("[covariate_model]: `{param}` is not assigned in [individual_parameters]")
    })?;
    let line = lines[idx].clone();
    let (lhs, rhs) = line
        .split_once('=')
        .expect("assignment_line matched an `=`");
    let rhs = rhs.trim();

    let mut rebuilt_rhs = if factors.is_empty() {
        rhs.to_string()
    } else {
        multiply_in(param, rhs, factors, eta_kappa)?
    };
    if !terms.is_empty() {
        // `if (c) a else b + t` puts the term inside the `else` arm, and
        // `a + b * f` is not `(a + b) * f`: anything that is not already a
        // plain top-level product has to be parenthesised before a term is
        // appended to it. A product needs no parentheses, and not adding them
        // keeps the generated line diffable against the hand-written one.
        if !is_plain_product(&rebuilt_rhs) {
            rebuilt_rhs = format!("({rebuilt_rhs})");
        }
        rebuilt_rhs.push_str(" + ");
        rebuilt_rhs.push_str(&terms.join(" + "));
    }
    lines[idx] = format!("{lhs}= {rebuilt_rhs}");
    Ok(())
}

/// Whether `rhs` is a top-level product of plain multiplicative terms — the
/// shape a factor can be inserted into, and the shape a term can be appended
/// to without parentheses.
fn is_plain_product(rhs: &str) -> bool {
    split_top_level(rhs, '*')
        .iter()
        .all(|p| is_simple_factor(p))
}

/// `factors` multiplied into `rhs` immediately before its first η/κ-bearing
/// top-level factor (or at the end when it has none).
fn multiply_in(
    param: &str,
    rhs: &str,
    factors: &[String],
    eta_kappa: &HashSet<String>,
) -> Result<String, String> {
    let parts = split_top_level(rhs, '*');
    // *Every* factor has to be a plain multiplicative term, not just the whole
    // RHS when it happens to carry no top-level `*`. Checking only the latter
    // let `CL = TVCL * exp(ETA_CL) + BASE_CL` through — `split_top_level` returns
    // two parts there, so the sum guard was skipped and the covariate factor was
    // multiplied into the first addend alone. `is_simple_factor` rejects an empty
    // part too, which covers a trailing/doubled `*`.
    if parts.iter().any(|p| !is_simple_factor(p)) {
        return Err(non_product_error(param, rhs));
    }
    let first_random = parts
        .iter()
        .position(|p| mentions_any(p, eta_kappa))
        .unwrap_or(parts.len());
    if first_random == 0 && parts.len() == 1 {
        // The whole RHS is one random-effect-bearing term (`exp(TVCL + ETA_CL)`);
        // there is no non-η group to multiply into.
        return Err(non_product_error(param, rhs));
    }

    let mut rebuilt: Vec<String> = parts.iter().map(|p| p.trim().to_string()).collect();
    for (offset, factor) in factors.iter().enumerate() {
        rebuilt.insert(first_random + offset, factor.clone());
    }
    Ok(rebuilt.join(" * "))
}

fn non_product_error(param: &str, rhs: &str) -> String {
    format!(
        "[covariate_model]: cannot place a covariate factor in `{param} = {rhs}` — the \
         right-hand side is not a top-level product, so multiplying into it would change what \
         the expression means (and, for a sum or a transform, would move the effect out of the \
         non-η group the SAEM mu-reference detector reads, #619).\n\
         An additive relation (a trailing `+`) has no such requirement — it is appended to the \
         right-hand side rather than multiplied into it — so this applies to a multiplicative \
         relation only.\n\
         Wire the factor by hand instead:\n\
         \x20 [individual_parameters]\n\
         \x20   {param} = ... * COV_{param} * ...   # COV_{param} = the factor, placed where you \
         want it"
    )
}

/// Whether a single-factor RHS is a plain multiplicative term (`TVCL`,
/// `exp(ETA_CL)`, `(WT/70)^0.75`) rather than a sum or a conditional that only
/// *looks* like one factor because it has no top-level `*`.
fn is_simple_factor(rhs: &str) -> bool {
    let rhs = rhs.trim();
    if rhs.is_empty() {
        return false;
    }
    // A leading `if` is a conditional / mixture branch, never a factor.
    if rhs.starts_with("if ") || rhs.starts_with("if(") {
        return false;
    }
    // A top-level `+` or `-` makes it a sum. Signs inside exponents (`1e-3`)
    // and a leading unary sign are not top-level operators.
    !has_top_level_additive(rhs)
}

// ── Text helpers ───────────────────────────────────────────────────────────

/// The `[individual_parameters]` names assigned at top level (brace depth 0),
/// in order. A name assigned inside an `if { ... }` block is deliberately not
/// listed: the block has no single right-hand side to multiply into.
fn top_level_assignments(lines: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    for line in lines {
        let text = strip_comment(line);
        if depth == 0 {
            if let Some(name) = assignment_target(text) {
                if !out.contains(&name) {
                    out.push(name);
                }
            }
        }
        depth += text.matches('{').count() as i32;
        depth -= text.matches('}').count() as i32;
        depth = depth.max(0);
    }
    out
}

/// The index of the top-level line assigning `param`.
fn assignment_line(lines: &[String], param: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut found = None;
    for (i, line) in lines.iter().enumerate() {
        let text = strip_comment(line);
        if depth == 0 && assignment_target(text).as_deref() == Some(param) {
            found = Some(i);
        }
        depth += text.matches('{').count() as i32;
        depth -= text.matches('}').count() as i32;
        depth = depth.max(0);
    }
    found
}

/// `CL = ...` → `Some("CL")`. Not an assignment if the `=` is part of a
/// comparison, or the left side is not a bare identifier.
fn assignment_target(line: &str) -> Option<String> {
    let (lhs, rhs) = line.split_once('=')?;
    if rhs.starts_with('=') || lhs.ends_with(['=', '<', '>', '!']) {
        return None;
    }
    let lhs = lhs.trim();
    if lhs.is_empty() || !is_identifier(lhs) {
        return None;
    }
    Some(lhs.to_string())
}

fn is_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Split on `sep` at parenthesis/bracket depth 0.
fn split_top_level(text: &str, sep: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for c in text.chars() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            _ => {}
        }
        if c == sep && depth == 0 {
            out.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    out.push(current);
    out
}

/// Whether the text carries a `+` or `-` at depth 0 that is a binary operator —
/// i.e. not a leading unary sign and not the sign of an exponent (`1e-3`).
fn has_top_level_additive(text: &str) -> bool {
    let bytes: Vec<char> = text.chars().collect();
    let mut depth = 0i32;
    for (i, c) in bytes.iter().enumerate() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            '+' | '-' if depth == 0 => {
                let prev = bytes[..i].iter().rposition(|c| !c.is_whitespace());
                let Some(prev) = prev else { continue }; // leading unary sign
                let p = bytes[prev];
                // `1e-3` / `1E+3`: the sign belongs to the exponent.
                if (p == 'e' || p == 'E') && prev > 0 && bytes[prev - 1].is_ascii_digit() {
                    continue;
                }
                // An operator can only follow a value, never another operator.
                if matches!(p, '+' | '-' | '*' | '/' | '^' | '(' | ',') {
                    continue;
                }
                return true;
            }
            _ => {}
        }
    }
    false
}

/// Whether any identifier in `text` is in `names`.
fn mentions_any(text: &str, names: &HashSet<String>) -> bool {
    let mut current = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            current.push(c);
        } else if !current.is_empty() {
            if names.contains(&current) {
                return true;
            }
            current.clear();
        }
    }
    !current.is_empty() && names.contains(&current)
}

/// Drop a trailing `# comment`, returning the text before it.
fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => line[..i].trim(),
        None => line.trim(),
    }
}

/// Render a number for emission into model text: integral values without the
/// `.0` so a generated line reads like a hand-written one, and everything else
/// with enough digits to round-trip.
fn fmt(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// Covariate levels as they would be written in a `levels = [...]` clause.
fn join_levels(levels: &[f64]) -> String {
    if levels.is_empty() {
        "none".to_string()
    } else {
        levels
            .iter()
            .map(|v| fmt(*v))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn join_names(names: &[String]) -> String {
    if names.is_empty() {
        "none".to_string()
    } else {
        names.join(", ")
    }
}

/// A one-edit-away suggestion for a misspelled form name.
fn did_you_mean(word: &str) -> String {
    let lower = word.to_lowercase();
    match FORM_SPELLINGS
        .iter()
        .filter(|f| levenshtein(&lower, f) <= 2)
        .min_by_key(|f| levenshtein(&lower, f))
    {
        Some(best) => format!(" (did you mean `{best}`?)"),
        None => String::new(),
    }
}

/// Every form the block accepts, in the order the diagnostics list them.
///
/// The two "expected one of …" messages and [`did_you_mean`] all read this one
/// array. They used to carry their own copies, and #1312 added `categorical2`
/// to two of the three — the missing-form diagnostic kept listing eight forms
/// and quietly hid the new one from anyone who left the form off. A single list
/// removes that failure mode rather than fixing one instance of it;
/// `every_form_the_parser_accepts_is_offered_by_both_diagnostics` pins that it
/// stays complete.
const FORM_SPELLINGS: &[&str] = &[
    "none",
    "linear",
    "linear_relative",
    "exponential",
    "power",
    "hockey",
    "categorical",
    "categorical2",
    "expr",
];

/// `FORM_SPELLINGS` as the diagnostics render it.
fn form_list() -> String {
    FORM_SPELLINGS.join(", ")
}

/// Levenshtein distance, for the did-you-mean above only.
fn levenshtein(a: &str, b: &str) -> usize {
    let b_chars: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
    let mut current = vec![0usize; b_chars.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        current[0] = i + 1;
        for (j, cb) in b_chars.iter().enumerate() {
            let cost = usize::from(ca != *cb);
            current[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(current[j] + 1);
        }
        std::mem::swap(&mut prev, &mut current);
    }
    prev[b_chars.len()]
}

#[cfg(test)]
#[path = "covariate_model_tests.rs"]
mod tests;
