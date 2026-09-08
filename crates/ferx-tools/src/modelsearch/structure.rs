//! The structural coordinates a candidate lives at, and how a set of them
//! becomes a template line (#1181, #1257).
//!
//! A structural PK model is five coordinates: **absorption** (bolus,
//! first-order, zero-order or Weibull), **elimination** (first-order,
//! zero-order, Michaelis-Menten or mixed), **peripheral compartments** (0–2),
//! **transit compartments** (none, a fixed count, or an estimated one) and
//! **lag time** (on or off). A [`Structure`] is one point in that space; a
//! [`FeatureKey`] is one MFL feature — one move along one axis — and applying
//! it to a structure gives the neighbouring point.
//!
//! Every point maps to exactly one template and role list
//! ([`Structure::template`]), or to none, in which case the search never
//! generates it and says so. The mapping is the coverage table
//! `docs/tools/modelsearch.qmd` publishes.
//!
//! # Two engines
//!
//! The analytic `pk NAME(...)` templates cover `ABSORPTION(INST | FO)` with
//! first-order elimination, and nothing else. `ABSORPTION(ZO)`,
//! `ABSORPTION(WEIBULL)` and every other `ELIMINATION` are reachable only as
//! an `ode_template NAME(...)` line plus one `[odes]` override — which is
//! what [`Structure::engine`] reports and `ferx-core::edit`'s
//! [`ferx_core::edit::StructuralEngine`] writes (#1257). A candidate that needs the ODE
//! engine costs one to two orders of magnitude more per fit than its analytic
//! siblings, so it also carries its own runtime weight and start budget; see
//! [`Structure::cost`] and [`Structure::starts`].
//!
//! The rules for which move is allowed from which point are Pharmpy's
//! (`modelsearch/algorithms.py`, `_is_allowed`), and they are checked against
//! Pharmpy's own enumeration in `pharmpy_anchor.rs`. Where ferx goes further
//! — a candidate that no template can express — the reason is a named
//! [`Structure::unbuildable`] string rather than a silent gap.

use std::collections::HashMap;
use std::fmt;

use ferx_core::edit::{EliminationForm, InputForm, NewParameter, StructuralSpec};
use ferx_core::Population;

use crate::search::mfl::{
    AbsorptionMode, DepotMode, EliminationMode, Feature, LagtimeMode, Mfl, Mode as _, TransitCounts,
};
use crate::search::{FeatureVector, PkTemplate};

/// The absorption a candidate has.
///
/// `INST` is the `*_iv` family and `FO` the `*_oral` one (and `*_transit`
/// when there are transit compartments) — both analytic. `ZO` and `WEIBULL`
/// have no analytic template at all: each is an `[odes]` input function on an
/// `*_iv` disposition (#1257). Pharmpy's `SEQ-ZO-FO` is not here, because it
/// is not a coordinate ferx builds; the coverage check refuses it by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Absorption {
    Inst,
    Fo,
    Zo,
    Weibull,
}

impl Absorption {
    pub fn label(&self) -> &'static str {
        match self {
            Absorption::Inst => "INST",
            Absorption::Fo => "FO",
            Absorption::Zo => "ZO",
            Absorption::Weibull => "WEIBULL",
        }
    }

    /// Whether the dose is delivered by an `[odes]` input function rather
    /// than by the template itself.
    fn needs_ode(&self) -> bool {
        matches!(self, Absorption::Zo | Absorption::Weibull)
    }
}

/// The elimination a candidate has — Pharmpy's `ELIMINATION` (#1257).
///
/// Only `FO` has an analytic template. The three saturable forms are the one
/// parameterisation Pharmpy uses (`CLMM·KM·C/(KM + C)`, i.e. `Vmax = CLMM·KM`),
/// written by [`ferx_core::pk::ode_template`]; `ZO` is `MM` with the Michaelis
/// constant fixed far below the observed concentrations, which is also how
/// Pharmpy writes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum Elimination {
    #[default]
    Fo,
    Zo,
    Mm,
    MixFoMm,
}

impl Elimination {
    pub fn label(&self) -> &'static str {
        match self {
            Elimination::Fo => "FO",
            Elimination::Zo => "ZO",
            Elimination::Mm => "MM",
            Elimination::MixFoMm => "MIX-FO-MM",
        }
    }

    /// Whether the flux out of `central` needs an `[odes]` override.
    fn needs_ode(&self) -> bool {
        !matches!(self, Elimination::Fo)
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
    Elimination(Elimination),
    Peripherals(u32),
    Transits(TransitCount),
    Lagtime(bool),
}

impl FeatureKey {
    /// The MFL keyword: the category the feature moves along.
    pub fn category(&self) -> &'static str {
        match self {
            FeatureKey::Absorption(_) => "ABSORPTION",
            FeatureKey::Elimination(_) => "ELIMINATION",
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
            FeatureKey::Elimination(e) => e.label().to_string(),
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
    /// First-order unless the candidate moved along `ELIMINATION` (#1257).
    pub elimination: Elimination,
    /// Peripheral compartments, 0–2.
    pub peripherals: u32,
    /// `None` when the drug is absorbed first-order (or not at all).
    pub transits: Option<TransitCount>,
    pub lagtime: bool,
}

/// The template line a structure renders to: name, the roles it takes, and
/// which engine writes the disposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Template {
    pub name: &'static str,
    /// Every role the structure needs, in the order they are written — the
    /// disposition roles, then absorption, then a lag time, then the
    /// saturable-elimination parameters.
    ///
    /// The last group and `dur` / `td` / `beta` are **not** bindings: an
    /// `ode_template` line takes only the analytic signature's roles, and
    /// these are read by the `[odes]` override instead ([`ODE_ONLY_ROLES`]).
    /// They are listed here so one loop derives a name, an init and an η for
    /// every parameter a candidate introduces, whichever engine reads it.
    pub roles: Vec<&'static str>,
    /// `Pk` for the analytic closed form, `Ode` for a candidate whose
    /// absorption or elimination has no `pk` template (#1257).
    pub engine: Engine,
}

/// Which engine a [`Structure`] needs — the tool-side mirror of
/// [`ferx_core::edit::StructuralEngine`], without the parameter names an edit
/// carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Engine {
    Pk,
    Ode,
}

/// Roles that a template line cannot carry: the `[odes]` override reads them
/// instead. An `ode_template NAME(...)` line takes exactly the analytic
/// `pk NAME(...)` signature (`crate::pk::ode_template::generate` rejects
/// anything else), so the zero-order duration, the Weibull pair and the
/// saturable-elimination pair are declared as parameters and referenced only
/// by the generated equation.
pub const ODE_ONLY_ROLES: &[&str] = &["dur", "td", "beta", "clmm", "km"];

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
    /// Read the structure off a `pk NAME(...)` line alone. A transit chain
    /// reads as `TRANSITS(N)`; see [`from_model`](Self::from_model) for the
    /// count the model actually declares.
    pub fn from_template(template: &PkTemplate) -> Result<Structure, String> {
        Self::from_model(template, None)
    }

    /// Read the structure off a base model's `pk NAME(...)` line and, for a
    /// transit chain, the declaration its `n` binding points at: a literal
    /// (`n=3`) or a parameter behind a `FIX`ed θ is `TRANSITS(3)`, a
    /// parameter behind a free θ is `TRANSITS(N)`.
    ///
    /// The `*_ig` templates and any `ode_template` line are refused: the
    /// former has no MFL coordinate, the latter carries its absorption and
    /// elimination in an `[odes]` block this cannot read back, so neither is
    /// a point the search can move from.
    pub fn from_model(
        template: &PkTemplate,
        text: Option<&ferx_core::edit::ModelText>,
    ) -> Result<Structure, String> {
        let name = template.name.as_str();
        if template.keyword != "pk" {
            // Refused *here*, before the input model is fitted: the failure
            // would otherwise come from `SetStructural`'s own guard, i.e. only
            // after the expensive step this check exists to save (#1256).
            //
            // The search *writes* `ode_template` candidates (#1257) but cannot
            // *read* one as a starting point: what an ODE candidate's
            // absorption and elimination are is stated by its `[odes]`
            // override, and recovering a coordinate from an arbitrary
            // right-hand side is a different problem from generating one.
            return Err(format!(
                "modelsearch: the base model's disposition is an `{kw} {name}(...)` line, whose \
                 absorption and elimination live in its `[odes]` block — the search reads its \
                 starting coordinates off a `pk NAME(...)` template line and cannot recover them \
                 from an equation. Write the base with `pk {name}(...)` to search over it \
                 (the search will write `ode_template` candidates of its own)",
                kw = template.keyword
            ));
        }
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
            "transit" => (
                Absorption::Fo,
                Some(match text {
                    Some(text) => transit_count_of(template, text)?,
                    None => TransitCount::N,
                }),
            ),
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
            // A `pk` line eliminates first-order; there is no other analytic
            // template, so this is the only elimination a base can carry.
            elimination: Elimination::Fo,
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
            FeatureKey::Elimination(e) => next.elimination = *e,
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
            FeatureKey::Elimination(self.elimination),
            FeatureKey::Peripherals(self.peripherals),
            FeatureKey::Transits(self.transits.unwrap_or(TransitCount::Count(0))),
            FeatureKey::Lagtime(self.lagtime),
        ]
    }

    /// Which engine writes this structure's disposition (#1257).
    pub fn engine(&self) -> Engine {
        if self.absorption.needs_ode() || self.elimination.needs_ode() {
            Engine::Ode
        } else {
            Engine::Pk
        }
    }

    /// The candidate's expected cost relative to an analytic sibling, for the
    /// runner's thread plan ([`Candidate::cost`](crate::search::Candidate)).
    ///
    /// An ODE candidate is integrated numerically where its analytic sibling
    /// has a closed form, which is the order-of-magnitude difference #1181
    /// declined to mix into one uniform plan.
    ///
    /// These are **coarse tiers, not measurements**: the runner groups
    /// candidates of equal cost and runs the heaviest group first, so only
    /// the ordering and the equality are used — never the magnitude. The
    /// measurement is the report's `seconds` column. For scale, one FOCEI
    /// evaluation of the warfarin Michaelis-Menten candidate takes 0.064 s
    /// against the analytic base's 0.008 s (`tests/nonmem/modelsearch_anchor`,
    /// at the tight tolerances the anchor pins), and a *fit* widens that:
    /// every outer iteration integrates the system again, with sensitivities.
    pub fn cost(&self) -> f64 {
        match self.engine() {
            Engine::Pk => 1.0,
            // A saturable right-hand side is stiffer than the linear one, so
            // a candidate carrying both is its own tier above one carrying
            // only an ODE input function.
            Engine::Ode if self.elimination.needs_ode() => 40.0,
            Engine::Ode => 20.0,
        }
    }

    /// The start budget this candidate needs, when it needs more than the
    /// run's ([`Candidate::starts`](crate::search::Candidate)).
    ///
    /// Saturable elimination is the canonical local-minimum case —
    /// `docs/examples/multistart.qmd` is a Michaelis-Menten model — and a
    /// search that ranks a stalled MM fit against a converged first-order one
    /// rejects a correct model on the strength of the optimizer. The extra
    /// starts are spent only on the candidates that need them.
    pub fn starts(&self) -> Option<usize> {
        self.elimination.needs_ode().then_some(MM_STARTS)
    }

    /// The candidate's coordinates for the runner table and the journal.
    pub fn feature_vector(&self) -> FeatureVector {
        self.features()
            .iter()
            .map(|k| (k.category().to_string(), k.argument()))
            .collect()
    }

    /// Why no template can express this point, when none can.
    ///
    /// Pharmpy's "not supported" pairs, applied to the *whole* structure
    /// rather than only to the features a path applied — so a base that
    /// already carries a lag time cannot be given transit compartments
    /// either. Plus one ferx-only gap: there is no `three_cpt_transit`.
    ///
    /// Elimination is unconstrained: it appears in none of Pharmpy's pairs,
    /// and the `[odes]` override that writes it replaces only the `central`
    /// equation, so it composes with every absorption and every compartment
    /// count.
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
        // Pharmpy's `not_supported_combo`: `ABSORPTION(ZO)` and
        // `ABSORPTION(WEIBULL)` each model the whole absorption delay
        // themselves, so a transit chain in front of one is not a candidate;
        // Weibull additionally does not take a lag time (its own `td` shifts
        // the profile), while zero-order does — Pharmpy lags the infusion.
        if self.absorption.needs_ode() && transits {
            return Some(format!(
                "{} absorption models the whole absorption delay itself, so it does not take a \
                 transit chain in front of it",
                self.absorption.label()
            ));
        }
        if self.absorption == Absorption::Weibull && self.lagtime {
            return Some(
                "Weibull absorption already carries its own delay through `td`; Pharmpy does \
                 not combine it with a lag time and neither does ferx"
                    .into(),
            );
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

    /// The template, role list and engine for this point.
    ///
    /// The roles are ordered disposition → absorption → lag time →
    /// elimination, and the last group (plus `dur` / `td` / `beta`) is read
    /// by the `[odes]` override rather than bound on the line — see
    /// [`ODE_ONLY_ROLES`].
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
        // Zero-order and Weibull absorption feed `central` directly, so their
        // disposition is the bolus template — the depot and its `ka` are gone,
        // exactly as Pharmpy's `set_zero_order_absorption` removes them.
        let (route, absorption): (&str, &[&str]) = match (self.absorption, self.transits) {
            (Absorption::Inst, _) => ("iv", &[]),
            (Absorption::Fo, None) => ("oral", &["ka"]),
            (Absorption::Fo, Some(_)) => ("transit", &["n", "mtt"]),
            (Absorption::Zo, _) => ("iv", &["dur"]),
            (Absorption::Weibull, _) => ("iv", &["td", "beta"]),
        };
        let mut roles: Vec<&'static str> = disposition.to_vec();
        roles.extend_from_slice(absorption);
        if self.lagtime {
            roles.push("lagtime");
        }
        // `MM` and `ZO` reuse the template's own `cl` binding as the
        // Michaelis-Menten clearance (Pharmpy renames `CL` to `CLMM`; ferx
        // keeps the name, and with it the parameter's estimate and its η), so
        // only the Michaelis constant is new. `MIX-FO-MM` keeps `CL` as a
        // first-order clearance and adds a saturable one beside it.
        roles.extend_from_slice(match self.elimination {
            Elimination::Fo => &[],
            Elimination::Zo | Elimination::Mm => &["km"],
            Elimination::MixFoMm => &["clmm", "km"],
        });
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
        Ok(Template {
            name,
            roles,
            engine: self.engine(),
        })
    }
}

// ── The space ───────────────────────────────────────────────────────────────

/// The structural features a resolved MFL space names, one [`FeatureKey`]
/// per (category, value), in Pharmpy's dictionary order — sorted by
/// `(category, str(argument))`.
///
/// The coverage check (#1179) has already refused every mode ferx cannot
/// build, so what reaches here is `INST`/`FO`/`ZO`/`WEIBULL`, every
/// `ELIMINATION`, `PERIPHERALS(0..2)`, `TRANSITS(n | N, NODEPOT)` and
/// `LAGTIME`. `ELIMINATION(FO)` contributes a key like any other value, so a
/// space naming it can move *back* to first-order elimination from a
/// saturable candidate.
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
                        AbsorptionMode::Zo => push(FeatureKey::Absorption(Absorption::Zo)),
                        AbsorptionMode::Weibull => {
                            push(FeatureKey::Absorption(Absorption::Weibull))
                        }
                        other => {
                            return Err(format!(
                                "modelsearch: ABSORPTION({}) has neither a `pk` template nor an \
                                 `[odes]` input function; the coverage check should have refused \
                                 it",
                                other.label()
                            ))
                        }
                    }
                }
            }
            Feature::Elimination(modes) => {
                for m in modes.expand() {
                    push(FeatureKey::Elimination(match m {
                        EliminationMode::Fo => Elimination::Fo,
                        EliminationMode::Zo => Elimination::Zo,
                        EliminationMode::Mm => Elimination::Mm,
                        EliminationMode::MixFoMm => Elimination::MixFoMm,
                    }));
                }
            }
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
    // Pharmpy's order: absorption, elimination, transits, peripherals, lagtime.
    for category in [
        "ABSORPTION",
        "ELIMINATION",
        "TRANSITS",
        "PERIPHERALS",
        "LAGTIME",
    ] {
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
/// are `funcs` (the base's own features already removed). `current` is the
/// structure the path stands on.
pub fn allowed(
    key: &FeatureKey,
    applied: &[FeatureKey],
    funcs: &[FeatureKey],
    current: &Structure,
) -> bool {
    if applied.contains(key) {
        return false;
    }
    // `TRANSITS(0)` is a move only *off* a chain: on a first-order model it
    // is the model itself (Pharmpy never allows it, since its base cannot
    // carry a chain into the space; ferx's can).
    if *key == FeatureKey::Transits(TransitCount::Count(0)) && current.transits.is_none() {
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
/// lag time or transits, `ABSORPTION(ZO)` or `ABSORPTION(WEIBULL)` with
/// transits, `ABSORPTION(WEIBULL)` with a lag time, and a lag time with
/// transits.
///
/// `ELIMINATION` appears in none of them: Pharmpy combines every elimination
/// with every absorption, and so does ferx. `SEQ-ZO-FO` is in Pharmpy's table
/// twice and in none of ferx's, because it is not a coordinate ferx builds at
/// all — the coverage check refuses it before a path is enumerated.
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
                | (Absorption(self::Absorption::Zo), Transits(_))
                | (Absorption(self::Absorption::Weibull), Transits(_))
                | (Absorption(self::Absorption::Weibull), Lagtime(true))
                | (Lagtime(true), Transits(_))
        )
    };
    pair(a, b) || pair(b, a)
}

/// Whether an exhaustive combination — at most one feature per category —
/// is one Pharmpy would build and ferx can, from `base`. The same pair
/// table as [`allowed`], applied to the combination itself.
pub fn combination_allowed(combo: &[FeatureKey], base: &Structure) -> bool {
    if combo.contains(&FeatureKey::Transits(TransitCount::Count(0))) && base.transits.is_none() {
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
/// parameter names and initial estimates, the dataset's first observation
/// time, and — for the saturable eliminations — its observation range.
///
/// The rules are Pharmpy's (`modeling/odes.py`), stated on the docs page:
/// a first peripheral gets `Q = CL`, `V2 = 0.05·Vc`; a second `Q3 = 0.9·CL`
/// (and a first added alongside it `0.1·CL`), `V3 = 0.05·Vc`; a lag time
/// and a mean transit time start at half the first positive observation
/// time; an estimated transit count at 2; an absorption rate at
/// `1 / (2·t_first)`.
///
/// The #1257 additions are Pharmpy's too: a zero-order input duration is
/// `2·MAT` with `MAT = 2·t_first`; a Weibull scale is `MAT / Γ(1 + 1/β)` at
/// `β = 1.5`; a Michaelis constant starts at `max(DV)/2` bounded above by
/// `1.5·max(DV)`, or is **fixed** at `min(DV)/100` for zero-order
/// elimination; and a mixed model's saturable clearance starts at `CL/2`.
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
    /// The largest observation in the data; `1.0` without data — Pharmpy's
    /// own fallback when a model carries no dataset.
    pub dv_max: f64,
    /// The smallest observation in the data; `1.0` without data. Only a
    /// zero-order elimination reads it, and it floors at Pharmpy's `0.01`
    /// when the data would put the fixed Michaelis constant at or below zero.
    pub dv_min: f64,
}

/// One `theta NAME(init, …)` declaration of a model text: its name, init and
/// whether it is `FIX`ed.
#[derive(Debug, Clone, PartialEq)]
pub struct ThetaDecl {
    pub name: String,
    pub init: f64,
    pub fixed: bool,
}

/// The `theta` declarations of a model text, read off the text rather than
/// the parse — so a candidate seeded from its parent's estimates scales its
/// new parameters from those, as Pharmpy's `update_initial_estimates` →
/// `add_peripheral_compartment` order does. A vector θ (`NAME[...]`) has no
/// single init and is skipped.
pub fn theta_decls_of(text: &ferx_core::edit::ModelText) -> Vec<ThetaDecl> {
    let mut out = Vec::new();
    for line in text.block_lines("parameters") {
        let Some(rest) = line.strip_prefix("theta ") else {
            continue;
        };
        let Some((name, args)) = rest.split_once('(') else {
            continue;
        };
        let name = name.trim();
        if name.contains('[') {
            continue;
        }
        if let Some(init) = args
            .split(',')
            .next()
            .and_then(|a| a.trim().trim_end_matches(')').trim().parse::<f64>().ok())
        {
            let fixed = args
                .split(|c: char| !c.is_alphanumeric())
                .any(|tok| tok.eq_ignore_ascii_case("FIX"));
            out.push(ThetaDecl {
                name: name.to_string(),
                init,
                fixed,
            });
        }
    }
    out
}

/// The `theta` inits of a model text, by name.
pub fn theta_inits_of(text: &ferx_core::edit::ModelText) -> HashMap<String, f64> {
    theta_decls_of(text)
        .into_iter()
        .map(|d| (d.name, d.init))
        .collect()
}

/// The η names a model text declares, on `omega` lines and inside
/// `block_omega (…)` headers.
pub fn eta_names_of(text: &ferx_core::edit::ModelText) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.block_lines("parameters") {
        if let Some(rest) = line.strip_prefix("omega ") {
            if let Some(name) = rest.split('~').next() {
                out.push(name.trim().to_string());
            }
        } else if let Some(rest) = line.strip_prefix("block_omega") {
            if let Some(inner) = rest.split_once('(').and_then(|(_, r)| r.split_once(')')) {
                out.extend(
                    inner
                        .0
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty()),
                );
            }
        }
    }
    out
}

/// The `[individual_parameters]` names a model text declares.
pub fn parameter_names_of(text: &ferx_core::edit::ModelText) -> Vec<String> {
    text.block_lines("individual_parameters")
        .iter()
        .filter_map(|l| l.split_once('=').map(|(lhs, _)| lhs.trim().to_string()))
        .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_alphanumeric() || c == '_'))
        .collect()
}

/// The transit count a `*_transit` line's `n` binding declares: a literal,
/// or a parameter set from a number or from a θ — `FIX`ed θ and literals
/// are a count, a free θ is `N`.
fn transit_count_of(
    template: &PkTemplate,
    text: &ferx_core::edit::ModelText,
) -> Result<TransitCount, String> {
    let var = template
        .bindings
        .iter()
        .find(|(role, _)| role == "n")
        .map(|(_, v)| v.as_str())
        .ok_or_else(|| {
            format!(
                "modelsearch: `pk {}(...)` binds no `n`; a transit template needs its count",
                template.name
            )
        })?;
    let count = |v: f64, what: &str| -> Result<TransitCount, String> {
        if v.is_finite() && v >= 0.0 && v.fract() == 0.0 {
            Ok(TransitCount::Count(v as u32))
        } else {
            Err(format!(
                "modelsearch: the transit count {what} is {v}, which is not a whole number; \
                 a fixed count must be an integer, or estimate it (`TRANSITS(N)`)"
            ))
        }
    };
    if let Ok(v) = var.parse::<f64>() {
        return count(v, &format!("`n={var}`"));
    }
    let rhs = text
        .block_lines("individual_parameters")
        .iter()
        .find_map(|l| {
            l.split_once('=')
                .filter(|(lhs, _)| lhs.trim() == var)
                .map(|(_, rhs)| rhs.trim().to_string())
        })
        .ok_or_else(|| {
            format!(
                "modelsearch: `n={var}` names `{var}`, which [individual_parameters] does not \
                 declare"
            )
        })?;
    if let Ok(v) = rhs.parse::<f64>() {
        return count(v, &format!("`{var} = {rhs}`"));
    }
    let thetas = theta_decls_of(text);
    let behind = rhs
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .find_map(|tok| thetas.iter().find(|d| d.name == tok));
    match behind {
        Some(d) if d.fixed => count(d.init, &format!("`theta {}` (FIX)", d.name)),
        _ => Ok(TransitCount::N),
    }
}

impl Defaults {
    /// The defaults read off one model text — the parent a candidate is
    /// derived from, *after* its own edits and seeding, so a name the input
    /// declared but an earlier step pruned is not taken as present, and a
    /// θ the parent added is seen as taken.
    pub fn of_text(&self, text: &ferx_core::edit::ModelText) -> Defaults {
        let thetas = theta_decls_of(text);
        Defaults {
            parameters: parameter_names_of(text),
            theta_names: thetas.iter().map(|d| d.name.clone()).collect(),
            eta_names: eta_names_of(text),
            theta_init: thetas.into_iter().map(|d| (d.name, d.init)).collect(),
            // The data-derived values belong to the run, not to the text.
            t_first: self.t_first,
            dv_max: self.dv_max,
            dv_min: self.dv_min,
        }
    }

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
        // Pharmpy reads `max(DV)` / `min(DV)` over the PK observations
        // (`_get_pk_observations`) and falls back to 1.0 when the model
        // carries no dataset; a population with no finite observation is the
        // same situation.
        let observations = || {
            population
                .subjects
                .iter()
                .flat_map(|s| s.observations.iter().copied())
                .filter(|v| v.is_finite())
        };
        let dv_max = observations().fold(f64::NEG_INFINITY, f64::max);
        let dv_min = observations().fold(f64::INFINITY, f64::min);
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
            dv_max: if dv_max.is_finite() { dv_max } else { 1.0 },
            dv_min: if dv_min.is_finite() { dv_min } else { 1.0 },
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

/// The start budget a saturable-elimination candidate asks for (#1257).
///
/// `docs/examples/multistart.qmd` puts Michaelis-Menten elimination at the
/// top of its local-minimum table and recommends 4–8 starts;
/// `examples/mm_multistart.ferx` uses 8. A search ranks candidates against
/// each other, so a stalled MM fit is not a slow answer but a **wrong** one —
/// it rejects a model on the strength of the optimizer. The budget is a
/// floor, not a replacement: a run configured with more `retries` keeps them.
pub const MM_STARTS: usize = 8;

/// Pharmpy's Weibull shape at the start (`set_weibull_absorption`'s `init_k`).
const WEIBULL_SHAPE_INIT: f64 = 1.5;

/// `Γ(1 + 1/1.5) = Γ(5/3)`, the mean of a unit-scale Weibull at
/// [`WEIBULL_SHAPE_INIT`].
///
/// Pharmpy starts the scale at `MAT / Γ(1 + 1/k)`, so that the initial
/// profile has the mean absorption time the first-order model would have had.
/// The shape is fixed at 1.5 there, so this is a constant rather than a
/// gamma-function call; `weibull_scale_matches_the_gamma_identity` pins it.
const GAMMA_1_PLUS_1_OVER_SHAPE: f64 = 0.9027452929509336;

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
        // The `[odes]` roles (#1257). `DUR` is ferx's spelling of Pharmpy's
        // `D1 = 2·MAT`; `TD` / `BETA` are its `LAMBDA` / `K`; `CLMM` and `KM`
        // are its own names.
        "dur" => "DUR",
        "td" => "TD",
        "beta" => "BETA",
        "clmm" => "CLMM",
        "km" => "KM",
        _ => unreachable!("only absorption/peripheral/elimination roles are ever new"),
    }
}

/// Build the edit that moves `parent` (whose template line is
/// `parent_template`) to `target`.
///
/// Roles the parent already binds keep their variable — by slot, so a
/// two-compartment `v1=V` satisfies a one-compartment `v`; roles it does
/// not get the default name and a [`NewParameter`] with the Pharmpy-derived
/// init, unless the model already declares a parameter of that name, which
/// the edit layer then binds as it is.
///
/// The [`ODE_ONLY_ROLES`] take the same treatment except that they are not
/// *bindings*: they are declared and then read by the `[odes]` override the
/// [`ferx_core::edit::StructuralEngine::Ode`] variant writes (#1257). Reuse works the same way
/// — a parent that already declares `KM` keeps it, and with it whatever the
/// seeding step put in its θ.
pub fn structural_spec(
    target: &Structure,
    parent: &Structure,
    parent_template: &PkTemplate,
    parent_lines: &[String],
    defaults: &Defaults,
    iiv: IivStrategy,
) -> Result<StructuralSpec, String> {
    let template = target.template()?;
    // A different transit coordinate needs a different `n` declaration —
    // another count, or fixed ↔ estimated — and the edit layer binds an
    // existing name as it is. So the count is bound to a *fresh* parameter,
    // and the parent's old one, now unreferenced, is pruned with its θ.
    let rebind_n = target.transits.is_some() && target.transits != parent.transits;
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
    // The variable chosen for each `ODE_ONLY_ROLES` role, so the variant
    // below can name it.
    let mut ode_names: HashMap<&str, String> = HashMap::new();
    for role in &template.roles {
        let ode_only = ODE_ONLY_ROLES.contains(role);
        let rebinding = *role == "n" && rebind_n;
        // An ODE-only role is never on the parent's template line, so the
        // "keep the parent's variable" branch cannot apply to it; reuse of an
        // existing declaration happens by name, below.
        if let Some(var) = bound.get(slot_of(role)).filter(|_| !rebinding && !ode_only) {
            bindings.push((role.to_string(), var.to_string()));
            continue;
        }
        let name = if rebinding {
            // Not the parent's own `n` variable (which must go), and not a
            // name the parent declares for anything else.
            let old = bound.get("n").copied().unwrap_or("");
            (1..)
                .map(|k| {
                    if k == 1 {
                        "NTR".to_string()
                    } else {
                        format!("NTR{k}")
                    }
                })
                .find(|n| n != old && !defaults.parameters.contains(n))
                .expect("an untaken name exists")
        } else {
            default_name(role).to_string()
        };
        if ode_only {
            ode_names.insert(role, name.clone());
        } else {
            bindings.push((role.to_string(), name.clone()));
        }
        if !rebinding && defaults.parameters.contains(&name) {
            // Declared in the parent already (unbound): the edit binds it —
            // or, for an ODE-only role, the override reads it — as it is.
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
            // Pharmpy's `_add_zero_order_absorption`: the infusion lasts
            // `2·MAT`, and a `MAT` the model does not already have starts at
            // twice the first observation time.
            "dur" => (4.0 * defaults.t_first, 0.0, 1e6),
            // …and its `set_weibull_absorption`: the scale is the same `MAT`
            // divided by `Γ(1 + 1/k)`, so the initial profile has that mean
            // absorption time.
            "td" => (2.0 * defaults.t_first / GAMMA_1_PLUS_1_OVER_SHAPE, 0.0, 1e6),
            "beta" => (WEIBULL_SHAPE_INIT, 0.0, 1e6),
            // `_get_mm_inits`: the saturable clearance of a *mixed* model
            // starts at half the first-order one it is added beside.
            "clmm" => (0.5 * cl, 0.0, 1e6),
            // `_do_michaelis_menten_elimination`: `KM` starts at half the
            // largest observation, bounded above by 1.5× it, with Pharmpy's
            // cap for the NONMEM-specific overflow case. Zero-order
            // elimination is that model with `KM` **fixed** far below the
            // observed concentrations, so the flux is `CLMM·KM` — a constant
            // — wherever the data live (`set_zero_order_elimination`).
            "km" => {
                let init = if target.elimination == Elimination::Zo {
                    defaults.dv_min / 100.0
                } else if defaults.dv_max / 2.0 >= 1e6 {
                    5e5
                } else {
                    defaults.dv_max / 2.0
                };
                (init, 0.0, 1.5 * defaults.dv_max)
            }
            _ => unreachable!(),
        };
        let fixed = (*role == "n" && matches!(target.transits, Some(TransitCount::Count(_))))
            || (*role == "km" && target.elimination == Elimination::Zo);
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
        let mut p = NewParameter::new(name, theta, init, lower, upper);
        if let Some((eta, variance)) = iiv {
            p = p.with_iiv(eta, variance);
        }
        if fixed {
            p = p.fixed();
        }
        new_parameters.push(p);
    }
    // Bioavailability is not a coordinate of the search, and every template
    // reads it: a parent's `f=` binding is carried as it is, or the swap
    // would silently reset F to 1 and prune its θ and η.
    if let Some(var) = bound.get("f") {
        bindings.push(("f".to_string(), var.to_string()));
    }
    let spec = StructuralSpec::new(template.name.to_string(), bindings, new_parameters);
    if template.engine == Engine::Pk {
        return Ok(spec);
    }
    // The variant, named from the parameters chosen above. `expect` is not a
    // guess: `template()` puts each of these roles in `roles` for exactly the
    // coordinate that reads it, and the loop records every ODE-only role it
    // walks.
    let named = |role: &str| -> String {
        ode_names
            .get(role)
            .cloned()
            .expect("template() lists the role its variant reads")
    };
    let input = match target.absorption {
        Absorption::Zo => InputForm::ZeroOrder { dur: named("dur") },
        Absorption::Weibull => InputForm::Weibull {
            td: named("td"),
            beta: named("beta"),
        },
        Absorption::Inst | Absorption::Fo => InputForm::Template,
    };
    let elimination = match target.elimination {
        Elimination::Fo => EliminationForm::FirstOrder,
        // Zero-order elimination differs from Michaelis-Menten only in the
        // `FIX` on `KM`, which is a `[parameters]` declaration, not a term.
        Elimination::Zo | Elimination::Mm => EliminationForm::MichaelisMenten { km: named("km") },
        Elimination::MixFoMm => EliminationForm::MixedFoMm {
            clmm: named("clmm"),
            km: named("km"),
        },
    };
    Ok(spec.ode(input, elimination))
}

#[cfg(test)]
#[path = "structure_tests.rs"]
mod tests;
