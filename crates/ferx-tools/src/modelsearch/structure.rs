//! The structural coordinates a candidate lives at, and how a set of them
//! becomes a `pk` line (#1181).
//!
//! A structural PK model, as far as the analytic templates can express one,
//! is four coordinates: **absorption** (bolus or first-order), **peripheral
//! compartments** (0–2), **transit compartments** (none, a fixed count, or
//! an estimated one) and **lag time** (on or off). A [`Structure`] is one
//! point in that space; a [`FeatureKey`] is one MFL feature — one move along
//! one axis — and applying it to a structure gives the neighbouring point.
//!
//! Every point maps to exactly one `pk` template and role list
//! ([`Structure::template`]), or to none, in which case the search never
//! generates it and says so. The mapping is the coverage table
//! `docs/tools/modelsearch.qmd` publishes.
//!
//! The rules for which move is allowed from which point are Pharmpy's
//! (`modelsearch/algorithms.py`, `_is_allowed`), and they are checked against
//! Pharmpy's own enumeration in `pharmpy_anchor.rs`. Where ferx goes further
//! — a candidate that no template can express — the reason is a named
//! [`Structure::unbuildable`] string rather than a silent gap.

use std::collections::HashMap;
use std::fmt;

use ferx_core::edit::{NewParameter, StructuralSpec};
use ferx_core::Population;

use crate::search::mfl::{
    AbsorptionMode, DepotMode, Feature, LagtimeMode, Mfl, Mode as _, TransitCounts,
};
use crate::search::{FeatureVector, PkTemplate};

/// The absorption route a template family has: `INST` (the `*_iv` family)
/// or `FO` (`*_oral`, and `*_transit` when there are transit compartments).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Absorption {
    Inst,
    Fo,
}

impl Absorption {
    pub fn label(&self) -> &'static str {
        match self {
            Absorption::Inst => "INST",
            Absorption::Fo => "FO",
        }
    }
}

/// `TRANSITS(N, NODEPOT)` — estimated count — or `TRANSITS(n, NODEPOT)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransitCount {
    /// The count is a continuous parameter (`theta TVNTR(2.0, …)`).
    N,
    /// A fixed count (`theta TVNTR(3.0, …) FIX`).
    Count(u32),
}

impl fmt::Display for TransitCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransitCount::N => f.write_str("N"),
            TransitCount::Count(n) => write!(f, "{n}"),
        }
    }
}

/// One MFL feature of the structural space — Pharmpy's `FeatureKey`,
/// restricted to what ferx can build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureKey {
    Absorption(Absorption),
    Peripherals(u32),
    Transits(TransitCount),
    Lagtime(bool),
}

impl FeatureKey {
    /// The MFL keyword: the category the feature moves along.
    pub fn category(&self) -> &'static str {
        match self {
            FeatureKey::Absorption(_) => "ABSORPTION",
            FeatureKey::Peripherals(_) => "PERIPHERALS",
            FeatureKey::Transits(_) => "TRANSITS",
            FeatureKey::Lagtime(_) => "LAGTIME",
        }
    }

    /// The feature's argument as Pharmpy prints it — what its keys are
    /// sorted on (`str(key[1])`), so `TRANSITS(N)` sorts after the counts.
    pub fn argument(&self) -> String {
        match self {
            FeatureKey::Absorption(a) => a.label().to_string(),
            FeatureKey::Peripherals(n) => n.to_string(),
            FeatureKey::Transits(t) => t.to_string(),
            FeatureKey::Lagtime(on) => if *on { "ON" } else { "OFF" }.to_string(),
        }
    }

    /// Pharmpy's sort key for a feature dictionary: `(category, str(arg))`.
    pub fn sort_key(&self) -> (&'static str, String) {
        (self.category(), self.argument())
    }
}

impl fmt::Display for FeatureKey {
    /// Pharmpy's `key_to_str`: `PERIPHERALS(1)`, `TRANSITS(3, NODEPOT)`,
    /// `LAGTIME(ON)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeatureKey::Transits(t) => write!(f, "TRANSITS({t}, NODEPOT)"),
            other => write!(f, "{}({})", other.category(), other.argument()),
        }
    }
}

/// One point of the structural space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Structure {
    pub absorption: Absorption,
    /// Peripheral compartments, 0–2.
    pub peripherals: u32,
    /// `None` when the drug is absorbed first-order (or not at all).
    pub transits: Option<TransitCount>,
    pub lagtime: bool,
}

/// The `pk` line a structure renders to: template name and the roles it
/// takes, in the order they are written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Template {
    pub name: &'static str,
    pub roles: Vec<&'static str>,
}

/// Roles that share a PK slot, so a base model's binding for one is the
/// binding for the other: `v`/`v1`, `q`/`q2`, `lagtime`/`alag`.
fn slot_of(role: &str) -> &str {
    match role {
        "v1" => "v",
        "q2" => "q",
        "alag" => "lagtime",
        other => other,
    }
}

impl Structure {
    /// Read the structure off a base model's `pk NAME(...)` line.
    ///
    /// The `*_ig` templates and any `ode_template` line are refused: the
    /// former has no MFL coordinate, the latter no analytic sibling to swap
    /// to, so neither is a point the search can move from.
    pub fn from_template(template: &PkTemplate) -> Result<Structure, String> {
        let name = template.name.as_str();
        let (cpt, rest) = name
            .split_once("_cpt_")
            .or_else(|| name.split_once("_compartment_"))
            .ok_or_else(|| {
                format!(
                    "modelsearch: `pk {name}(...)` is not a `<n>_cpt_<route>` template the \
                     search can move from"
                )
            })?;
        let peripherals = match cpt {
            "one" => 0,
            "two" => 1,
            "three" => 2,
            _ => {
                return Err(format!(
                    "modelsearch: `pk {name}(...)` names an unknown compartment count `{cpt}`"
                ))
            }
        };
        let (absorption, transits) = match rest {
            "iv" => (Absorption::Inst, None),
            "oral" => (Absorption::Fo, None),
            "transit" => (Absorption::Fo, Some(TransitCount::N)),
            "ig" => {
                return Err(format!(
                    "modelsearch: the base model's `pk {name}(...)` uses inverse-Gaussian \
                     absorption, which has no MFL coordinate (ABSORPTION takes INST, FO, ZO, \
                     SEQ-ZO-FO and WEIBULL); start the search from a `*_oral` or `*_iv` base"
                ))
            }
            _ => {
                return Err(format!(
                    "modelsearch: `pk {name}(...)` names an unknown route `{rest}`"
                ))
            }
        };
        let lagtime = template
            .bindings
            .iter()
            .any(|(role, _)| slot_of(role) == "lagtime");
        Ok(Structure {
            absorption,
            peripherals,
            transits,
            lagtime,
        })
    }

    /// The point one feature away.
    pub fn apply(&self, key: &FeatureKey) -> Structure {
        let mut next = *self;
        match key {
            FeatureKey::Absorption(a) => next.absorption = *a,
            FeatureKey::Peripherals(n) => next.peripherals = *n,
            FeatureKey::Transits(TransitCount::Count(0)) => next.transits = None,
            FeatureKey::Transits(t) => next.transits = Some(*t),
            FeatureKey::Lagtime(on) => next.lagtime = *on,
        }
        next
    }

    /// The feature this structure carries in each category, as Pharmpy's
    /// `get_model_features` would list it — what the space is filtered
    /// against, so a feature the base already has is never a candidate move.
    pub fn features(&self) -> Vec<FeatureKey> {
        vec![
            FeatureKey::Absorption(self.absorption),
            FeatureKey::Peripherals(self.peripherals),
            FeatureKey::Transits(self.transits.unwrap_or(TransitCount::Count(0))),
            FeatureKey::Lagtime(self.lagtime),
        ]
    }

    /// The candidate's coordinates for the runner table and the journal.
    pub fn feature_vector(&self) -> FeatureVector {
        self.features()
            .iter()
            .map(|k| (k.category().to_string(), k.argument()))
            .collect()
    }

    /// Why no analytic template can express this point, when none can.
    ///
    /// Pharmpy's "not supported" pairs, applied to the *whole* structure
    /// rather than only to the features a path applied — so a base that
    /// already carries a lag time cannot be given transit compartments
    /// either. Plus one ferx-only gap: there is no `three_cpt_transit`.
    pub fn unbuildable(&self) -> Option<String> {
        let transits = self.transits.is_some();
        if self.absorption == Absorption::Inst && transits {
            return Some(
                "transit compartments need first-order absorption; there is no bolus \
                 template with a transit chain"
                    .into(),
            );
        }
        if self.absorption == Absorption::Inst && self.lagtime {
            return Some("a lag time on a bolus dose is not a structural candidate".into());
        }
        if self.lagtime && transits {
            return Some(
                "a lag time and a transit chain both model absorption delay; Pharmpy does \
                 not combine them and neither does ferx"
                    .into(),
            );
        }
        if transits && self.peripherals >= 2 {
            return Some(
                "there is no `three_cpt_transit` template: transit absorption is analytic \
                 for one and two compartments only"
                    .into(),
            );
        }
        if self.peripherals > 2 {
            return Some("the analytic templates stop at `three_cpt_*`".into());
        }
        None
    }

    /// The `pk` template and role list for this point.
    pub fn template(&self) -> Result<Template, String> {
        if let Some(why) = self.unbuildable() {
            return Err(why);
        }
        let disposition: &[&str] = match self.peripherals {
            0 => &["cl", "v"],
            1 => &["cl", "v1", "q", "v2"],
            _ => &["cl", "v1", "q2", "v2", "q3", "v3"],
        };
        let cpt = ["one", "two", "three"][self.peripherals as usize];
        let (route, absorption): (&str, &[&str]) = match (self.absorption, self.transits) {
            (Absorption::Inst, _) => ("iv", &[]),
            (Absorption::Fo, None) => ("oral", &["ka"]),
            (Absorption::Fo, Some(_)) => ("transit", &["n", "mtt"]),
        };
        let mut roles: Vec<&'static str> = disposition.to_vec();
        roles.extend_from_slice(absorption);
        if self.lagtime {
            roles.push("lagtime");
        }
        let name: &'static str = match (cpt, route) {
            ("one", "iv") => "one_cpt_iv",
            ("one", "oral") => "one_cpt_oral",
            ("one", "transit") => "one_cpt_transit",
            ("two", "iv") => "two_cpt_iv",
            ("two", "oral") => "two_cpt_oral",
            ("two", "transit") => "two_cpt_transit",
            ("three", "iv") => "three_cpt_iv",
            ("three", "oral") => "three_cpt_oral",
            _ => unreachable!("unbuildable() rejects every other combination"),
        };
        Ok(Template { name, roles })
    }
}

// ── The space ───────────────────────────────────────────────────────────────

/// The structural features a resolved MFL space names, one [`FeatureKey`]
/// per (category, value), in Pharmpy's dictionary order — sorted by
/// `(category, str(argument))`.
///
/// The coverage check (#1179) has already refused every mode ferx cannot
/// build, so what reaches here is `INST`/`FO`, `PERIPHERALS(0..2)`,
/// `TRANSITS(n | N, NODEPOT)` and `LAGTIME`. `ELIMINATION(FO)` is accepted
/// and contributes nothing: every template eliminates first-order.
pub fn space_features(mfl: &Mfl) -> Result<Vec<FeatureKey>, String> {
    let mut keys: Vec<FeatureKey> = Vec::new();
    let mut push = |k: FeatureKey| {
        if !keys.contains(&k) {
            keys.push(k);
        }
    };
    for feature in mfl.features() {
        match feature {
            Feature::Absorption(modes) => {
                for m in modes.expand() {
                    match m {
                        AbsorptionMode::Inst => push(FeatureKey::Absorption(Absorption::Inst)),
                        AbsorptionMode::Fo => push(FeatureKey::Absorption(Absorption::Fo)),
                        other => {
                            return Err(format!(
                                "modelsearch: ABSORPTION({}) has no analytic template; the \
                                 coverage check should have refused it",
                                other.label()
                            ))
                        }
                    }
                }
            }
            Feature::Elimination(_) => {}
            Feature::Peripherals { counts, .. } => {
                for n in counts.expand() {
                    push(FeatureKey::Peripherals(n));
                }
            }
            Feature::Transits { counts, depot } => {
                if depot
                    .as_ref()
                    .is_some_and(|d| d.expand().contains(&DepotMode::Depot))
                {
                    return Err(
                        "modelsearch: TRANSITS(n, DEPOT) has no analytic template; the coverage \
                         check should have refused it"
                            .into(),
                    );
                }
                match counts {
                    TransitCounts::N => push(FeatureKey::Transits(TransitCount::N)),
                    TransitCounts::Counts(c) => {
                        for n in c.expand() {
                            push(FeatureKey::Transits(TransitCount::Count(n)));
                        }
                    }
                }
            }
            Feature::Lagtime(modes) => {
                for m in modes.expand() {
                    push(FeatureKey::Lagtime(m == LagtimeMode::On));
                }
            }
            other => {
                return Err(format!(
                    "[space] mfl: `{}` is not a structural statement; modelsearch takes \
                     ABSORPTION, ELIMINATION, PERIPHERALS, TRANSITS and LAGTIME only. Covariate \
                     and variability features belong to covsearch, iivsearch and iovsearch, and \
                     ALLOMETRY to `ferx allometry` (#1175)",
                    other.keyword()
                ))
            }
        }
    }
    keys.sort_by_key(|k| k.sort_key());
    Ok(keys)
}

/// Pharmpy's `least_number_of_transformations`: the moves that take a base
/// lying outside the space onto it — for each category the space names
/// whose values do not include the base's, the space's first value (its
/// smallest count for `PERIPHERALS`).
pub fn onto_space(base: &Structure, space: &[FeatureKey]) -> Vec<FeatureKey> {
    let has = base.features();
    let mut moves = Vec::new();
    // Pharmpy's order: absorption, (elimination,) transits, peripherals, lagtime.
    for category in ["ABSORPTION", "TRANSITS", "PERIPHERALS", "LAGTIME"] {
        let listed: Vec<&FeatureKey> = space.iter().filter(|k| k.category() == category).collect();
        if listed.is_empty() || listed.iter().any(|k| has.contains(k)) {
            continue;
        }
        // Sorted order puts the smallest count first, and Pharmpy takes
        // `min(counts)` for peripherals and the first mode otherwise.
        moves.push(*listed[0]);
    }
    moves
}

/// Pharmpy's `_is_allowed`: whether `key` may be applied next on a path
/// that has already applied `applied`, in a space whose candidate features
/// are `funcs` (the base's own features already removed).
pub fn allowed(key: &FeatureKey, applied: &[FeatureKey], funcs: &[FeatureKey]) -> bool {
    if applied.contains(key) {
        return false;
    }
    if let FeatureKey::Peripherals(n) = key {
        // The first peripheral move must be the smallest count in the
        // space; a later one may be any other count. Pharmpy's rule, kept
        // verbatim so the enumeration anchors.
        let all: Vec<u32> = funcs
            .iter()
            .filter_map(|k| match k {
                FeatureKey::Peripherals(m) => Some(*m),
                _ => None,
            })
            .collect();
        let previous = applied
            .iter()
            .any(|k| matches!(k, FeatureKey::Peripherals(_)));
        if !previous {
            return all.iter().min() == Some(n);
        }
        let Some(index) = all.iter().position(|m| m == n) else {
            return false;
        };
        return index > 0 && all[index - 1] < *n;
    }
    // Equivalent to first-order absorption: never a move.
    if *key == FeatureKey::Transits(TransitCount::Count(0)) {
        return false;
    }
    if applied.iter().any(|k| k.category() == key.category()) {
        return false;
    }
    if applied.is_empty() {
        return true;
    }
    !applied.iter().any(|k| pharmpy_incompatible(key, k))
}

/// Pharmpy's `not_supported_combo` table, on the pairs ferx can name:
/// `ABSORPTION(FO)` with `TRANSITS(1, NODEPOT)`, `ABSORPTION(INST)` with a
/// lag time or transits, a lag time with transits.
fn pharmpy_incompatible(a: &FeatureKey, b: &FeatureKey) -> bool {
    use FeatureKey::*;
    let pair = |x: &FeatureKey, y: &FeatureKey| {
        matches!(
            (x, y),
            (
                Absorption(self::Absorption::Fo),
                Transits(TransitCount::Count(1))
            ) | (Absorption(self::Absorption::Inst), Lagtime(true))
                | (Absorption(self::Absorption::Inst), Transits(_))
                | (Lagtime(true), Transits(_))
        )
    };
    pair(a, b) || pair(b, a)
}

/// Whether an exhaustive combination — at most one feature per category —
/// is one Pharmpy would build and ferx can. The same pair table as
/// [`allowed`], applied to the combination itself.
pub fn combination_allowed(combo: &[FeatureKey]) -> bool {
    if combo.contains(&FeatureKey::Transits(TransitCount::Count(0))) {
        return false;
    }
    for (i, a) in combo.iter().enumerate() {
        for b in &combo[i + 1..] {
            if pharmpy_incompatible(a, b) {
                return false;
            }
        }
    }
    true
}

// ── From a structure to an edit ─────────────────────────────────────────────

/// What the new-parameter defaults are derived from: the base model's
/// parameter names and initial estimates, and the dataset's first
/// observation time.
///
/// The rules are Pharmpy's (`modeling/odes.py`), stated on the docs page:
/// a first peripheral gets `Q = CL`, `V2 = 0.05·Vc`; a second `Q3 = 0.9·CL`
/// (and a first added alongside it `0.1·CL`), `V3 = 0.05·Vc`; a lag time
/// and a mean transit time start at half the first positive observation
/// time; an estimated transit count at 2; an absorption rate at
/// `1 / (2·t_first)`.
#[derive(Debug, Clone)]
pub struct Defaults {
    /// `[individual_parameters]` names the base declares.
    pub parameters: Vec<String>,
    /// θ names already declared, so a generated `TVQ` does not collide.
    pub theta_names: Vec<String>,
    /// η names already declared.
    pub eta_names: Vec<String>,
    /// Initial estimates by θ name.
    pub theta_init: HashMap<String, f64>,
    /// The smallest positive observation time in the data; `1.0` without data.
    pub t_first: f64,
}

/// The `theta NAME(init, …)` inits of a model text, by name — read off the
/// text rather than the parse so a candidate seeded from its parent's
/// estimates scales its new parameters from those, as Pharmpy's
/// `update_initial_estimates` → `add_peripheral_compartment` order does.
pub fn theta_inits_of(text: &ferx_core::edit::ModelText) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    for line in text.block_lines("parameters") {
        let Some(rest) = line.strip_prefix("theta ") else {
            continue;
        };
        let Some((name, args)) = rest.split_once('(') else {
            continue;
        };
        let name = name.trim();
        // A vector θ (`NAME[...]`) has no single init.
        if name.contains('[') {
            continue;
        }
        if let Some(init) = args
            .split(',')
            .next()
            .and_then(|a| a.trim().trim_end_matches(')').trim().parse::<f64>().ok())
        {
            out.insert(name.to_string(), init);
        }
    }
    out
}

impl Defaults {
    /// Read the defaults off a base model and its dataset.
    pub fn new(
        parameters: Vec<String>,
        theta_names: Vec<String>,
        theta_init: Vec<f64>,
        eta_names: Vec<String>,
        population: &Population,
    ) -> Defaults {
        let t_first = population
            .subjects
            .iter()
            .flat_map(|s| s.obs_times.iter().copied())
            .filter(|t| *t > 0.0 && t.is_finite())
            .fold(f64::INFINITY, f64::min);
        Defaults {
            parameters,
            theta_init: theta_names
                .iter()
                .cloned()
                .zip(theta_init.iter().copied())
                .collect(),
            theta_names,
            eta_names,
            t_first: if t_first.is_finite() { t_first } else { 1.0 },
        }
    }

    /// The initial estimate behind an individual parameter: the first θ its
    /// `[individual_parameters]` line mentions. `None` when the line reads
    /// no declared θ (a parameter set from a covariate or a literal).
    fn init_behind(&self, lines: &[String], param: &str) -> Option<f64> {
        let line = lines.iter().find(|l| {
            l.split_once('=')
                .is_some_and(|(lhs, _)| lhs.trim() == param)
        })?;
        let rhs = line.split_once('=')?.1;
        rhs.split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .find_map(|tok| self.theta_init.get(tok).copied())
    }

    /// A θ name not yet taken: `TVQ`, or `TVQ_2` when the base has one.
    fn theta_name(&self, param: &str, taken: &[String]) -> String {
        let base = format!("TV{param}");
        if !self.theta_names.contains(&base) && !taken.contains(&base) {
            return base;
        }
        (2..)
            .map(|k| format!("{base}_{k}"))
            .find(|n| !self.theta_names.contains(n) && !taken.contains(n))
            .expect("an untaken suffix exists")
    }

    fn eta_name(&self, param: &str, taken: &[String]) -> String {
        let base = format!("ETA_{param}");
        if !self.eta_names.contains(&base) && !taken.contains(&base) {
            return base;
        }
        (2..)
            .map(|k| format!("{base}_{k}"))
            .find(|n| !self.eta_names.contains(n) && !taken.contains(n))
            .expect("an untaken suffix exists")
    }
}

/// How η is given to the parameters a candidate introduces — Pharmpy's
/// `iiv_strategy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IivStrategy {
    /// No η on any new parameter.
    NoAdd,
    /// A diagonal η (variance 0.01) on every new PK parameter.
    AddDiagonal,
    /// An η (variance 0.01) on the absorption-delay parameter only — a new
    /// lag time or mean transit time. Pharmpy's default.
    #[default]
    AbsorptionDelay,
    /// Pharmpy's `fullblock`: not available — `block_omega` over the new and
    /// existing η is a variability search's move (#1183). Refused at load.
    Fullblock,
}

impl IivStrategy {
    pub fn label(&self) -> &'static str {
        match self {
            IivStrategy::NoAdd => "no_add",
            IivStrategy::AddDiagonal => "add_diagonal",
            IivStrategy::AbsorptionDelay => "absorption_delay",
            IivStrategy::Fullblock => "fullblock",
        }
    }
}

/// Pharmpy's initial variance for an η added by the search.
pub const NEW_IIV_VARIANCE: f64 = 0.01;

/// Default variable names by role, for a role the parent does not bind.
fn default_name(role: &str) -> &'static str {
    match role {
        "ka" => "KA",
        "q" => "Q",
        "q2" => "Q",
        "v2" => "V2",
        "q3" => "Q3",
        "v3" => "V3",
        "n" => "NTR",
        "mtt" => "MTT",
        "lagtime" => "ALAG",
        _ => unreachable!("only absorption/peripheral roles are ever new"),
    }
}

/// Build the edit that moves `parent` (whose `pk` line is `parent_template`)
/// to `target`.
///
/// Roles the parent already binds keep their variable — by slot, so a
/// two-compartment `v1=V` satisfies a one-compartment `v`; roles it does
/// not get the default name and a [`NewParameter`] with the Pharmpy-derived
/// init, unless the model already declares a parameter of that name, which
/// the edit layer then binds as it is.
pub fn structural_spec(
    target: &Structure,
    parent_template: &PkTemplate,
    parent_lines: &[String],
    defaults: &Defaults,
    iiv: IivStrategy,
) -> Result<StructuralSpec, String> {
    let template = target.template()?;
    let bound: HashMap<&str, &str> = parent_template
        .bindings
        .iter()
        .map(|(role, var)| (slot_of(role), var.as_str()))
        .collect();

    let cl_init = bound
        .get("cl")
        .and_then(|p| defaults.init_behind(parent_lines, p));
    let vc_init = bound
        .get("v")
        .and_then(|p| defaults.init_behind(parent_lines, p));
    // Pharmpy's fallback when the model gives no clearance/volume to scale from.
    let cl = cl_init.unwrap_or(0.1);
    let vc = vc_init.unwrap_or(0.1 / 0.05);

    let mut bindings = Vec::with_capacity(template.roles.len());
    let mut new_parameters = Vec::new();
    let mut taken_theta: Vec<String> = Vec::new();
    let mut taken_eta: Vec<String> = Vec::new();
    for role in &template.roles {
        if let Some(var) = bound.get(slot_of(role)) {
            bindings.push((role.to_string(), var.to_string()));
            continue;
        }
        let name = default_name(role).to_string();
        bindings.push((role.to_string(), name.clone()));
        if defaults.parameters.contains(&name) {
            // Declared in the base already (unbound): the edit binds it as is.
            continue;
        }
        let (init, lower, upper) = match *role {
            "ka" => (1.0 / (2.0 * defaults.t_first), 0.0, 1e6),
            "q" | "q2" => (
                if target.peripherals >= 2 {
                    0.1 * cl
                } else {
                    cl
                },
                0.0,
                1e6,
            ),
            "q3" => (0.9 * cl, 0.0, 1e6),
            "v2" | "v3" => (0.05 * vc, 0.0, 1e6),
            "lagtime" | "mtt" => (defaults.t_first / 2.0, 0.0, 1e6),
            "n" => match target.transits {
                Some(TransitCount::Count(n)) => (n as f64, 0.0, 64.0),
                _ => (2.0, 0.0, 64.0),
            },
            _ => unreachable!(),
        };
        let fixed = *role == "n" && matches!(target.transits, Some(TransitCount::Count(_)));
        let delay = matches!(*role, "lagtime" | "mtt");
        let with_iiv = match iiv {
            IivStrategy::NoAdd => false,
            IivStrategy::AddDiagonal => !fixed,
            IivStrategy::AbsorptionDelay => delay,
            IivStrategy::Fullblock => {
                return Err(
                    "modelsearch: iiv_strategy = \"fullblock\" is not available (#1183)".into(),
                )
            }
        };
        let theta = defaults.theta_name(&name, &taken_theta);
        taken_theta.push(theta.clone());
        let iiv = with_iiv.then(|| {
            let eta = defaults.eta_name(&name, &taken_eta);
            taken_eta.push(eta.clone());
            (eta, NEW_IIV_VARIANCE)
        });
        new_parameters.push(NewParameter {
            name,
            theta,
            init,
            lower,
            upper,
            iiv,
            fixed,
        });
    }
    Ok(StructuralSpec {
        template: template.name.to_string(),
        bindings,
        new_parameters,
    })
}

#[cfg(test)]
#[path = "structure_tests.rs"]
mod tests;
