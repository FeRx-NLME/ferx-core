//! `@`-symbol resolution: turning `COVARIATE?(@IIV, @CONTINUOUS, *)` into the
//! explicit parameter × covariate × effect set it stands for on one base
//! model (#1179).
//!
//! The built-in symbols are Pharmpy's, resolved against the base model's
//! `[individual_parameters]`, its template line (`pk NAME(...)` or
//! `ode_template NAME(...)` — the same role grammar), its `[covariates]`
//! block and the dataset:
//!
//! | symbol | resolves to |
//! |---|---|
//! | `@IIV` | every individual parameter carrying an η |
//! | `@PK` | every individual parameter bound on the template line, in that order |
//! | `@PK_IIV` | `@PK` ∩ `@IIV` |
//! | `@ABSORPTION` | the `ka`, `n`, `mtt`, `mat`, `cv2` and `lagtime` bindings |
//! | `@ELIMINATION` | the `cl` binding |
//! | `@DISTRIBUTION` | the `v`/`v1`/`v2`/`v3` and `q`/`q2`/`q3` bindings |
//! | `@BIOAVAIL` | the `f` binding |
//! | `@CONTINUOUS` | the covariates declared `continuous` |
//! | `@CATEGORICAL` | the covariates declared `categorical` |
//! | `@PD`, `@PD_IIV` | nothing — there is no PD template family |
//!
//! `LET(NAME, [...])` overrides a built-in of the same name, so
//! `LET(CONTINUOUS, [WT, AGE])` narrows `@CONTINUOUS` without touching the
//! model file.
//!
//! Resolution is strict where it can be: an explicit name that is not an
//! individual parameter (or not a declared covariate) is an error naming it
//! and listing what would have been accepted, and a categorical effect on a
//! continuous covariate is refused here rather than by the candidate's parser
//! fifty fits later. A symbol that resolves to *nothing* drops the feature
//! with a note — Pharmpy's behaviour — because `COVARIATE?(@IIV,
//! @CATEGORICAL, cat)` on a dataset without categoricals is a space that
//! happens to be empty, not a mistake.

use std::collections::{HashMap, HashSet};

use ferx_core::edit::ModelText;
use ferx_core::{CovariateKind, ParsedModel, Population};

use super::mfl::{CovariateEffect, CovariateOp, Feature, Mfl, Mode, Modes, Operand, Statement};

/// The template line of a base model: `pk NAME(role=VAR, ...)` or
/// `ode_template NAME(role=VAR, ...)`. Both bind the same roles to the same
/// individual parameters; only the disposition's engine differs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PkTemplate {
    /// The keyword the line opened with: `pk` or `ode_template`. A search
    /// that *swaps* the line needs to know which — `ferx_core::edit`'s
    /// `SetStructural` writes a `pk` line and refuses any other — so the
    /// distinction cannot be dropped at parse time.
    pub keyword: String,
    /// The template name, e.g. `two_cpt_oral`.
    pub name: String,
    /// `(role, value)` in source order, e.g. `[("cl", "CL"), ("v", "V")]`.
    /// A value is whatever the line bound — usually an
    /// `[individual_parameters]` name, but the parser also accepts a numeric
    /// literal (`f=1`), so callers that want *parameters* intersect with the
    /// model's.
    pub bindings: Vec<(String, String)>,
}

impl PkTemplate {
    /// The keywords that open a template line.
    const KEYWORDS: &'static [&'static str] = &["pk", "ode_template"];

    /// Parse a `pk NAME(role=VAR, ...)` or `ode_template NAME(role=VAR, ...)`
    /// line. `None` when the line is neither.
    pub fn parse_line(line: &str) -> Option<Result<PkTemplate, String>> {
        let trimmed = line.trim();
        let (keyword, rest) = Self::KEYWORDS.iter().find_map(|kw| {
            let rest = trimmed.strip_prefix(kw)?;
            rest.starts_with(|c: char| c.is_whitespace())
                .then_some((*kw, rest))
        })?;
        let rest = rest.trim();
        let open = match rest.find('(') {
            Some(i) => i,
            None => return Some(Err(format!("`{keyword}` line has no `(`: `{line}`"))),
        };
        let name = rest[..open].trim().to_string();
        let close = match rest.rfind(')') {
            Some(i) if i > open => i,
            _ => {
                return Some(Err(format!(
                    "`{keyword}` line has no closing `)`: `{line}`"
                )))
            }
        };
        let mut bindings = Vec::new();
        for pair in rest[open + 1..close].split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            let Some((role, var)) = pair.split_once('=') else {
                return Some(Err(format!(
                    "`{keyword} {name}(...)`: `{pair}` is not a `role=VAR` binding"
                )));
            };
            bindings.push((role.trim().to_ascii_lowercase(), var.trim().to_string()));
        }
        Some(Ok(PkTemplate {
            keyword: keyword.to_string(),
            name,
            bindings,
        }))
    }

    fn with_roles(&self, roles: &[&str]) -> Vec<String> {
        let mut out = Vec::new();
        for (role, var) in &self.bindings {
            if roles.contains(&role.as_str()) && !out.contains(var) {
                out.push(var.clone());
            }
        }
        out
    }

    /// Every bound value, in template-line order.
    pub fn parameters(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (_, var) in &self.bindings {
            if !out.contains(var) {
                out.push(var.clone());
            }
        }
        out
    }
}

/// Roles of the `pk(...)` line, by Pharmpy's PK-parameter kinds.
const ABSORPTION_ROLES: &[&str] = &["ka", "n", "mtt", "mat", "cv2", "lagtime", "alag", "tlag"];
const ELIMINATION_ROLES: &[&str] = &["cl"];
const DISTRIBUTION_ROLES: &[&str] = &["v", "v1", "v2", "v3", "q", "q2", "q3"];
const BIOAVAIL_ROLES: &[&str] = &["f"];

/// What the symbols are resolved against: the parts of one base model and
/// its dataset that name parameters and covariates.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelContext {
    /// `[individual_parameters]` names, in source order.
    pub parameters: Vec<String>,
    /// The subset of `parameters` carrying an η, in η order.
    pub iiv: Vec<String>,
    /// The `pk` / `ode_template` line; `None` for an `ode(...)` or algebraic
    /// model, in which case the structural symbols are unavailable.
    pub template: Option<PkTemplate>,
    /// The `[covariates]` declarations; `None` when the block is absent.
    pub covariates: Option<Vec<(String, CovariateKind)>>,
    /// The dataset's covariate columns.
    pub data_covariates: Vec<String>,
}

impl ModelContext {
    /// Build the context from a parsed base model, its source text and its
    /// dataset.
    ///
    /// Checks the two things a search would otherwise discover per candidate:
    /// every declared covariate exists in the data, and the model names at
    /// least one individual parameter.
    pub fn from_model(
        parsed: &ParsedModel,
        text: &ModelText,
        population: &Population,
    ) -> Result<Self, String> {
        // The parser's list of top-level `[individual_parameters]`
        // assignments, in source order — not a text scan, which would read a
        // block-form `if (WT > 70) {` header as a parameter named `if (WT >`.
        // The parser also appends synthetic `__ferx_*` parameters (readouts,
        // direct `pk(...=TIME)` bindings); nobody searches over those.
        let parameters: Vec<String> = parsed
            .model
            .indiv_param_names
            .iter()
            .filter(|n| !n.starts_with("__ferx_"))
            .cloned()
            .collect();
        if parameters.is_empty() {
            return Err(
                "search space: the base model declares no `[individual_parameters]`, so there \
                 is nothing for `@PK`, `@IIV` or an explicit parameter name to refer to"
                    .into(),
            );
        }
        let mut iiv = Vec::new();
        for info in &parsed.model.eta_param_info {
            if !iiv.contains(&info.individual_param_name) {
                iiv.push(info.individual_param_name.clone());
            }
        }
        let mut template = None;
        for line in text.block_lines("structural_model") {
            if let Some(parsed_line) = PkTemplate::parse_line(&line) {
                template = Some(parsed_line?);
                break;
            }
        }
        let covariates = parsed.covariate_decls.as_ref().map(|decls| {
            decls
                .iter()
                .map(|d| (d.name.clone(), d.kind))
                .collect::<Vec<_>>()
        });
        let data_covariates = population.covariate_names.clone();
        if let Some(decls) = &covariates {
            let missing: Vec<&str> = decls
                .iter()
                .map(|(name, _)| name.as_str())
                .filter(|name| !data_covariates.iter().any(|c| c == name))
                .collect();
            if !missing.is_empty() && !data_covariates.is_empty() {
                return Err(format!(
                    "search space: `[covariates]` declares {} but the dataset has no such \
                     column; it has: {}",
                    missing
                        .iter()
                        .map(|m| format!("`{m}`"))
                        .collect::<Vec<_>>()
                        .join(", "),
                    data_covariates.join(", ")
                ));
            }
        }
        Ok(ModelContext {
            parameters,
            iiv,
            template,
            covariates,
            data_covariates,
        })
    }

    /// Every covariate a `COVARIATE` statement may name: the declared ones
    /// when there is a `[covariates]` block, else the dataset's columns.
    pub fn all_covariates(&self) -> Vec<String> {
        match &self.covariates {
            Some(decls) => decls.iter().map(|(n, _)| n.clone()).collect(),
            None => self.data_covariates.clone(),
        }
    }

    /// The declared kind of a covariate; `None` when undeclared (no block).
    pub fn covariate_kind(&self, name: &str) -> Option<CovariateKind> {
        self.covariates
            .as_ref()?
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, k)| *k)
    }

    fn declared_of_kind(&self, kind: CovariateKind, symbol: &str) -> Result<Vec<String>, String> {
        match &self.covariates {
            Some(decls) => Ok(decls
                .iter()
                .filter(|(_, k)| *k == kind)
                .map(|(n, _)| n.clone())
                .collect()),
            None => Err(format!(
                "search space: `@{symbol}` needs a `[covariates]` block in the base model to \
                 say which columns are {} — the dataset alone cannot tell a categorical \
                 covariate from a continuous one. Declare them, e.g. `WT continuous` / \
                 `SEX categorical(levels = [0, 1])`, or define the symbol with \
                 `LET({symbol}, [...])`",
                kind.label()
            )),
        }
    }

    /// The template's bound values that are individual parameters of the
    /// model, in template order and with the model's spelling. A numeric
    /// literal (`f=1`) or an unbound alias is not a parameter and is skipped,
    /// so `@BIOAVAIL` on such a line resolves to nothing rather than to a
    /// name the user never typed.
    fn template_parameters(&self, bound: Vec<String>) -> Vec<String> {
        let mut out = Vec::new();
        for value in bound {
            if let Some(p) = self
                .parameters
                .iter()
                .find(|p| p.eq_ignore_ascii_case(&value))
            {
                if !out.contains(p) {
                    out.push(p.clone());
                }
            }
        }
        out
    }

    fn no_template_error(&self, symbol: &str, let_name: &str) -> String {
        format!(
            "search space: `@{symbol}` reads the `pk NAME(...)` or `ode_template NAME(...)` \
             line, and the base model has none (an `ode(...)` or algebraic model). Name the \
             parameters explicitly, or define the symbol with `LET({let_name}, [...])`"
        )
    }

    fn template_roles(&self, symbol: &str, roles: &[&str]) -> Result<Vec<String>, String> {
        match &self.template {
            Some(t) => Ok(self.template_parameters(t.with_roles(roles))),
            None => Err(self.no_template_error(symbol, symbol)),
        }
    }

    /// Resolve a built-in symbol. Names are matched case-insensitively.
    pub fn builtin(&self, symbol: &str) -> Result<Vec<String>, String> {
        let upper = symbol.to_ascii_uppercase();
        match upper.as_str() {
            "IIV" => Ok(self.iiv.clone()),
            "PK" => match &self.template {
                Some(t) => Ok(self.template_parameters(t.parameters())),
                None => Err(self.no_template_error(symbol, "PK")),
            },
            "PK_IIV" => {
                let pk = self.builtin("PK")?;
                Ok(pk.into_iter().filter(|p| self.iiv.contains(p)).collect())
            }
            "PD" | "PD_IIV" => Ok(Vec::new()),
            "ABSORPTION" => self.template_roles(symbol, ABSORPTION_ROLES),
            "ELIMINATION" => self.template_roles(symbol, ELIMINATION_ROLES),
            "DISTRIBUTION" => self.template_roles(symbol, DISTRIBUTION_ROLES),
            "BIOAVAIL" => self.template_roles(symbol, BIOAVAIL_ROLES),
            "CONTINUOUS" => self.declared_of_kind(CovariateKind::Continuous, symbol),
            "CATEGORICAL" => self.declared_of_kind(CovariateKind::Categorical, symbol),
            _ => Err(format!(
                "search space: `@{symbol}` is not a built-in symbol and no `LET({symbol}, [...])` \
                 defines it; built-ins: @IIV, @PK, @PK_IIV, @PD, @PD_IIV, @ABSORPTION, \
                 @ELIMINATION, @DISTRIBUTION, @BIOAVAIL, @CONTINUOUS, @CATEGORICAL"
            )),
        }
    }
}

/// One fully explicit covariate effect the resolved space contains.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CovariateEffectSpec {
    pub parameter: String,
    pub covariate: String,
    pub effect: CovariateEffect,
    pub op: CovariateOp,
    /// `true` for `COVARIATE?` (the search may include it), `false` for
    /// `COVARIATE` (every candidate carries it).
    pub optional: bool,
}

impl CovariateEffectSpec {
    /// Short key for tables and feature vectors, e.g. `CL-WT`.
    pub fn pair(&self) -> String {
        format!("{}-{}", self.parameter, self.covariate)
    }
}

/// A space with every symbol and wildcard replaced by explicit names.
///
/// `mfl` is ground — [`Operand::Names`] and [`Modes::List`] only, `LET`s
/// removed — so its [`Mfl::render`] is a canonical description of the space
/// on this model. `covariate_effects` is the same information for the
/// `COVARIATE` statements, one tuple per effect, after Pharmpy's override
/// rules; the two are built from one list and cannot disagree.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    pub mfl: Mfl,
    pub covariate_effects: Vec<CovariateEffectSpec>,
    /// Features dropped because a symbol resolved to nothing, and other
    /// things a user should see once.
    pub notes: Vec<String>,
}

/// One ground `COVARIATE` statement's identity: (parameter, covariate,
/// operator, optional).
type CovariateGroupKey = (String, String, CovariateOp, bool);

/// Resolve every symbol and wildcard in `mfl` against `ctx`.
pub fn resolve(mfl: &Mfl, ctx: &ModelContext) -> Result<Resolved, String> {
    let lets = collect_lets(mfl)?;
    let mut notes = Vec::new();
    let mut statements = Vec::new();
    let mut covariate_terms: Vec<(Vec<CovariateEffectSpec>, bool)> = Vec::new();

    for feature in mfl.features() {
        match feature {
            Feature::Covariate {
                optional,
                parameters,
                covariates,
                effects,
                op,
            } => {
                // Pharmpy's override rule (covariate.py, `Covariate.eval`):
                // a statement is overridable only when its *parameter*
                // operand is a symbol. `COVARIATE?(CL, @CONTINUOUS, …)` names
                // CL explicitly, so a later `COVARIATE(CL, WT, exp)` adds to
                // it rather than replacing its CL-WT terms.
                let explicit = !parameters.is_symbolic();
                let params = resolve_parameters(parameters, ctx, &lets, &feature.keyword(), "PK")?;
                let covs = resolve_covariates(covariates, ctx, &lets, &feature.keyword())?;
                if params.is_empty() || covs.is_empty() {
                    notes.push(format!(
                        "dropped `{feature}`: {} resolves to nothing on this model",
                        if params.is_empty() {
                            parameters.to_string()
                        } else {
                            covariates.to_string()
                        }
                    ));
                    continue;
                }
                // Pharmpy's `product(parameter, covariate)` order: parameter
                // outermost, so the ground statements read CL first.
                let mut terms = Vec::new();
                for param in &params {
                    for cov in &covs {
                        let kind = ctx.covariate_kind(cov);
                        let effect_list: Vec<CovariateEffect> = match effects {
                            // Pharmpy expands `*` to the continuous forms; on a
                            // declared categorical covariate the only form that
                            // can apply is `cat`.
                            Modes::Wildcard => match kind {
                                Some(CovariateKind::Categorical) => vec![CovariateEffect::Cat],
                                _ => CovariateEffect::CONTINUOUS.to_vec(),
                            },
                            Modes::List(list) => list.clone(),
                        };
                        for effect in effect_list {
                            check_effect_kind(cov, kind, effect, feature)?;
                            terms.push(CovariateEffectSpec {
                                parameter: param.clone(),
                                covariate: cov.clone(),
                                effect,
                                op: *op,
                                optional: *optional,
                            });
                        }
                    }
                }
                covariate_terms.push((terms, explicit));
            }
            Feature::Iiv {
                optional,
                parameters,
                effects,
            }
            | Feature::Iov {
                optional,
                parameters,
                effects,
            } => {
                // `IIV(*, …)` is every parameter an η could go on; `IOV(*, …)`
                // is every parameter that already has one (Pharmpy's iovsearch
                // adds κ on top of the IIV set).
                let wildcard = if matches!(feature, Feature::Iiv { .. }) {
                    "PK"
                } else {
                    "IIV"
                };
                let params =
                    resolve_parameters(parameters, ctx, &lets, &feature.keyword(), wildcard)?;
                if params.is_empty() {
                    notes.push(format!(
                        "dropped `{feature}`: {parameters} resolves to nothing on this model"
                    ));
                    continue;
                }
                let effects = Modes::List(effects.expand());
                let ground = if matches!(feature, Feature::Iiv { .. }) {
                    Feature::Iiv {
                        optional: *optional,
                        parameters: Operand::Names(params),
                        effects,
                    }
                } else {
                    Feature::Iov {
                        optional: *optional,
                        parameters: Operand::Names(params),
                        effects,
                    }
                };
                statements.push(Statement::Feature(ground));
            }
            Feature::Covariance {
                optional,
                level,
                parameters,
            } => {
                // `COVARIANCE(IIV, *)`: the wildcard is the η-bearing set, not
                // `@PK` — a covariance between parameters without an η is not
                // a candidate anything can build.
                let params = resolve_parameters(parameters, ctx, &lets, &feature.keyword(), "IIV")?;
                if params.len() < 2 {
                    notes.push(format!(
                        "dropped `{feature}`: {parameters} resolves to {} parameter{} on this \
                         model, and a covariance needs two",
                        params.len(),
                        if params.len() == 1 { "" } else { "s" }
                    ));
                    continue;
                }
                statements.push(Statement::Feature(Feature::Covariance {
                    optional: *optional,
                    level: Modes::List(level.expand()),
                    parameters: Operand::Names(params),
                }));
            }
            Feature::Allometry { covariate, .. } => {
                check_covariate_name(covariate, ctx, "ALLOMETRY")?;
                statements.push(Statement::Feature(feature.clone()));
            }
            other => statements.push(Statement::Feature(ground_modes(other))),
        }
    }

    let covariate_effects = merge_covariate_terms(covariate_terms)?;
    // Rebuild the COVARIATE statements from the merged list so `mfl` and
    // `covariate_effects` are one thing: one statement per
    // (parameter, covariate, op, optional), effects in first-seen order.
    let mut groups: Vec<(CovariateGroupKey, Vec<CovariateEffect>)> = Vec::new();
    for t in &covariate_effects {
        let key = (t.parameter.clone(), t.covariate.clone(), t.op, t.optional);
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, effects)) => {
                if !effects.contains(&t.effect) {
                    effects.push(t.effect);
                }
            }
            None => groups.push((key, vec![t.effect])),
        }
    }
    for ((parameter, covariate, op, optional), effects) in groups {
        statements.push(Statement::Feature(Feature::Covariate {
            optional,
            parameters: Operand::Names(vec![parameter]),
            covariates: Operand::Names(vec![covariate]),
            effects: Modes::List(effects),
            op,
        }));
    }

    Ok(Resolved {
        mfl: Mfl { statements },
        covariate_effects,
        notes,
    })
}

fn collect_lets(mfl: &Mfl) -> Result<HashMap<String, Vec<String>>, String> {
    let mut lets: HashMap<String, Vec<String>> = HashMap::new();
    for statement in &mfl.statements {
        if let Statement::Let { name, values } = statement {
            let key = name.to_ascii_uppercase();
            if lets.contains_key(&key) {
                return Err(format!(
                    "search space: `LET({name}, ...)` is defined twice; a symbol has one \
                     definition"
                ));
            }
            lets.insert(key, values.clone());
        }
    }
    Ok(lets)
}

fn lookup(
    symbol: &str,
    ctx: &ModelContext,
    lets: &HashMap<String, Vec<String>>,
) -> Result<Vec<String>, String> {
    if let Some(values) = lets.get(&symbol.to_ascii_uppercase()) {
        return Ok(values.clone());
    }
    ctx.builtin(symbol)
}

fn check_parameter_name(name: &str, ctx: &ModelContext, context: &str) -> Result<(), String> {
    if ctx.parameters.iter().any(|p| p == name) {
        Ok(())
    } else {
        Err(format!(
            "search space: in {context}: `{name}` is not an individual parameter of the base \
             model; it has: {}",
            ctx.parameters.join(", ")
        ))
    }
}

fn check_covariate_name(name: &str, ctx: &ModelContext, context: &str) -> Result<(), String> {
    let known = ctx.all_covariates();
    if known.iter().any(|c| c == name) {
        return Ok(());
    }
    let where_ = if ctx.covariates.is_some() {
        "declared in the base model's `[covariates]` block"
    } else {
        "a covariate column of the dataset"
    };
    Err(format!(
        "search space: in {context}: `{name}` is not {where_}; known: {}",
        if known.is_empty() {
            "(none)".to_string()
        } else {
            known.join(", ")
        }
    ))
}

/// `wildcard` names the built-in symbol `*` stands for in this slot — the
/// wildcard is feature-specific in MFL (`@PK` for a covariate's parameters,
/// `@IIV` for a covariance's), so the caller says which. It resolves exactly
/// as `@{wildcard}` would, `LET` overrides included.
fn resolve_parameters(
    operand: &Operand,
    ctx: &ModelContext,
    lets: &HashMap<String, Vec<String>>,
    context: &str,
    wildcard: &str,
) -> Result<Vec<String>, String> {
    let names = match operand {
        Operand::Names(names) => names.clone(),
        Operand::Symbol(symbol) => lookup(symbol, ctx, lets)?,
        // `*` *is* the slot's built-in symbol, so a `LET` of that name
        // governs it too: `LET(PK,[CL]); IIV(*,EXP)` is `IIV(CL,EXP)`.
        Operand::Wildcard => lookup(wildcard, ctx, lets)?,
    };
    let mut out: Vec<String> = Vec::new();
    for name in names {
        check_parameter_name(&name, ctx, context)?;
        if !out.contains(&name) {
            out.push(name);
        }
    }
    Ok(out)
}

fn resolve_covariates(
    operand: &Operand,
    ctx: &ModelContext,
    lets: &HashMap<String, Vec<String>>,
    context: &str,
) -> Result<Vec<String>, String> {
    let names = match operand {
        Operand::Names(names) => names.clone(),
        Operand::Symbol(symbol) => lookup(symbol, ctx, lets)?,
        Operand::Wildcard => ctx.all_covariates(),
    };
    let mut out: Vec<String> = Vec::new();
    for name in names {
        check_covariate_name(&name, ctx, context)?;
        if !out.contains(&name) {
            out.push(name);
        }
    }
    Ok(out)
}

fn check_effect_kind(
    cov: &str,
    kind: Option<CovariateKind>,
    effect: CovariateEffect,
    feature: &Feature,
) -> Result<(), String> {
    match kind {
        Some(CovariateKind::Categorical) if !effect.is_categorical() => Err(format!(
            "search space: in `{feature}`: `{cov}` is declared categorical, and `{}` is a \
             continuous form; use `cat`",
            effect.label()
        )),
        Some(CovariateKind::Continuous) if effect.is_categorical() => Err(format!(
            "search space: in `{feature}`: `{cov}` is declared continuous, and `{}` is a \
             categorical form; use one of lin, piece_lin, exp, pow",
            effect.label()
        )),
        _ => Ok(()),
    }
}

/// Pharmpy's `expand()` rules over the per-statement covariate terms:
///
/// 1. a `(parameter, covariate)` pair from a statement whose *parameter*
///    operand is explicit wins over the same pair produced by a statement
///    whose parameter is a symbol or wildcard, which is dropped (Pharmpy
///    gates on the parameter operand alone; a symbolic covariate does not
///    make a statement overridable);
/// 2. a forced pair (`COVARIATE`) named by more than one statement is an
///    error — the space would be over-specified;
/// 3. a forced term wins over the identical optional term.
fn merge_covariate_terms(
    terms: Vec<(Vec<CovariateEffectSpec>, bool)>,
) -> Result<Vec<CovariateEffectSpec>, String> {
    let explicit_pairs: HashSet<(String, String)> = terms
        .iter()
        .filter(|(_, explicit)| *explicit)
        .flat_map(|(list, _)| {
            list.iter()
                .map(|t| (t.parameter.clone(), t.covariate.clone()))
        })
        .collect();

    let mut forced_owner: HashMap<(String, String), usize> = HashMap::new();
    let mut all: Vec<CovariateEffectSpec> = Vec::new();
    for (i, (list, explicit)) in terms.into_iter().enumerate() {
        for t in list {
            let pair = (t.parameter.clone(), t.covariate.clone());
            if !explicit && explicit_pairs.contains(&pair) {
                continue;
            }
            if !t.optional {
                match forced_owner.get(&pair) {
                    Some(owner) if *owner != i => {
                        return Err(format!(
                            "search space: the covariate effect {}-{} is forced by more than \
                             one COVARIATE statement; state it once",
                            t.parameter, t.covariate
                        ));
                    }
                    _ => {
                        forced_owner.insert(pair, i);
                    }
                }
            }
            all.push(t);
        }
    }

    let forced: HashSet<(String, String, CovariateEffect, CovariateOp)> = all
        .iter()
        .filter(|t| !t.optional)
        .map(|t| (t.parameter.clone(), t.covariate.clone(), t.effect, t.op))
        .collect();
    let mut out: Vec<CovariateEffectSpec> = Vec::new();
    for t in all {
        if t.optional
            && forced.contains(&(t.parameter.clone(), t.covariate.clone(), t.effect, t.op))
        {
            continue;
        }
        if !out.contains(&t) {
            out.push(t);
        }
    }
    Ok(out)
}

/// Expand every `*` mode of a structural / PD feature.
fn ground_modes(feature: &Feature) -> Feature {
    fn g<M: Mode>(m: &Modes<M>) -> Modes<M> {
        Modes::List(m.expand())
    }
    match feature {
        Feature::Absorption(m) => Feature::Absorption(g(m)),
        Feature::Elimination(m) => Feature::Elimination(g(m)),
        Feature::Peripherals { counts, kind } => Feature::Peripherals {
            counts: counts.clone(),
            kind: kind.as_ref().map(g),
        },
        Feature::Transits { counts, depot } => Feature::Transits {
            counts: counts.clone(),
            depot: depot.as_ref().map(g),
        },
        Feature::Lagtime(m) => Feature::Lagtime(g(m)),
        Feature::DirectEffect(m) => Feature::DirectEffect(g(m)),
        Feature::EffectComp(m) => Feature::EffectComp(g(m)),
        Feature::IndirectEffect { mode, production } => Feature::IndirectEffect {
            mode: g(mode),
            production: g(production),
        },
        Feature::Metabolite(m) => Feature::Metabolite(g(m)),
        other => other.clone(),
    }
}

#[cfg(test)]
#[path = "resolve_tests.rs"]
mod tests;
