//! What ferx can build out of a parsed MFL space — and the hard error for what
//! it cannot (#1179).
//!
//! The rule from the epic (#1175): a parsed feature ferx cannot express is a
//! **hard error naming the feature**, never a silently narrowed search. A
//! search that quietly drops `ELIMINATION(MM)` would report "FO elimination
//! wins" over a space that never contained the alternative, and nothing in the
//! report would say so.
//!
//! The table here is the one `docs/tools/search.qmd` publishes; every reason
//! string names the missing piece so the error is actionable, and every error
//! links to the table.

use super::mfl::{
    AbsorptionMode, CovariateEffect, CovariateOp, DepotMode, Feature, Mfl, Mode, Modes,
    PeripheralKind, TransitCounts, VariabilityEffect,
};

/// Where the coverage table lives, appended to every gap error.
pub const COVERAGE_DOCS: &str = "https://ferx-nlme.github.io/ferx-core/tools/search.html#coverage";

/// The most peripheral compartments an analytic `pk` template has
/// (`three_cpt_*`).
pub const MAX_PERIPHERALS: u32 = 2;

/// One feature (or one value of a feature) the space asks for and ferx has no
/// candidate for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gap {
    /// The offending feature as it would be written, e.g. `ELIMINATION(MM)`.
    pub feature: String,
    /// Why there is no candidate for it, and what would be needed.
    pub reason: String,
}

/// Every gap in a space, reported together so one round of editing fixes
/// them all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageError {
    pub gaps: Vec<Gap>,
}

impl std::fmt::Display for CoverageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "the search space asks for {} feature{} ferx cannot build a candidate for:",
            self.gaps.len(),
            if self.gaps.len() == 1 { "" } else { "s" }
        )?;
        for gap in &self.gaps {
            writeln!(f, "  - {}: {}", gap.feature, gap.reason)?;
        }
        write!(f, "See the coverage table at {COVERAGE_DOCS}")
    }
}

impl std::error::Error for CoverageError {}

impl From<CoverageError> for String {
    fn from(e: CoverageError) -> String {
        e.to_string()
    }
}

/// Check every feature of a (parsed, not necessarily resolved) space against
/// what ferx can express. `Ok(())` means every feature has a candidate.
///
/// Wildcards are checked against their full expansion, so
/// `ELIMINATION(*)` fails on `ZO`, `MM` and `MIX-FO-MM` just as the explicit
/// list would — a wildcard is not a way to opt out of the check.
pub fn check_coverage(mfl: &Mfl) -> Result<(), CoverageError> {
    let mut gaps = Vec::new();
    for feature in mfl.features() {
        check_feature(feature, &mut gaps);
    }
    if gaps.is_empty() {
        Ok(())
    } else {
        Err(CoverageError { gaps })
    }
}

fn gap(gaps: &mut Vec<Gap>, feature: String, reason: impl Into<String>) {
    let reason = reason.into();
    let g = Gap { feature, reason };
    if !gaps.contains(&g) {
        gaps.push(g);
    }
}

fn check_feature(feature: &Feature, gaps: &mut Vec<Gap>) {
    match feature {
        Feature::Absorption(modes) => {
            // `INST` and `FO` are the analytic `*_iv` / `*_oral` templates;
            // `ZO` and `WEIBULL` are the `[odes]` input functions
            // `zero_order(...)` and `weibull(...)` on an `*_iv` disposition,
            // which `ferx-core::edit` writes as an `ode_template` candidate
            // (#1257).
            //
            // `SEQ-ZO-FO` is the one that stays out. It is not a *term* on
            // the central equation like the other two: the dose fills a depot
            // at a constant rate and the depot empties first-order
            // (`examples/sequential_absorption.ferx`), which is a second
            // compartment, its own `ka` and its own duration — a different
            // disposition, not a different input. Generating it needs a
            // template family that does not exist rather than an override.
            for m in modes.expand() {
                match m {
                    AbsorptionMode::Inst
                    | AbsorptionMode::Fo
                    | AbsorptionMode::Zo
                    | AbsorptionMode::Weibull => {}
                    AbsorptionMode::SeqZoFo => gap(
                        gaps,
                        "ABSORPTION(SEQ-ZO-FO)".into(),
                        "sequential zero-order-then-first-order absorption is not one input term \
                         on a standard disposition but a depot of its own, filled at a constant \
                         rate and emptied by `ka`; there is no template for it and the search \
                         does not generate one. Write it by hand — \
                         `examples/sequential_absorption.ferx` — and search around it",
                    ),
                }
            }
        }
        // Every elimination is buildable (#1257): first-order is the analytic
        // template's own, and the three saturable forms are an `[odes]`
        // override of the `central` equation on the same disposition.
        Feature::Elimination(_) => {}
        Feature::Peripherals { counts, kind } => {
            for n in counts.expand() {
                if n > MAX_PERIPHERALS {
                    gap(
                        gaps,
                        format!("PERIPHERALS({n})"),
                        format!(
                            "the analytic templates stop at `three_cpt_*` \
                             ({MAX_PERIPHERALS} peripheral compartments)"
                        ),
                    );
                }
            }
            if let Some(kind) = kind {
                if kind.expand().contains(&PeripheralKind::Met) {
                    gap(
                        gaps,
                        "PERIPHERALS(n, MET)".into(),
                        "there is no metabolite template family (structsearch is out of scope \
                         for #1175 v1)",
                    );
                }
            }
        }
        Feature::Transits { counts, depot } => {
            // Pharmpy's defaults when the option is omitted: DEPOT for a
            // count, NODEPOT for `N` (its grammar gives `TRANSITS(N)` no
            // option at all). A Pharmpy-ported `TRANSITS(1..3)` therefore
            // asks for a depot, and must hit the gap rather than be quietly
            // read as the NODEPOT family ferx can build.
            let default = match counts {
                TransitCounts::N => DepotMode::NoDepot,
                TransitCounts::Counts(_) => DepotMode::Depot,
            };
            let modes = depot
                .as_ref()
                .map(Modes::expand)
                .unwrap_or_else(|| vec![default]);
            if modes.contains(&DepotMode::Depot) {
                gap(
                    gaps,
                    "TRANSITS(n, DEPOT)".into(),
                    "the analytic `*_transit` templates feed central straight from the last \
                     transit compartment (Pharmpy's NODEPOT); a separate depot with its own \
                     `ka` needs an `[odes]` model with `transit(...) - KA*depot`. Write \
                     NODEPOT explicitly — `TRANSITS(n)` alone means DEPOT, as in Pharmpy",
                );
            }
        }
        Feature::Lagtime(_) => {}
        Feature::Covariate { effects, op, .. } => {
            // The covariate wildcard is Pharmpy's four continuous forms, not
            // the full enum — `cat2` and `custom` have to be asked for by name.
            let effects = match effects {
                Modes::Wildcard => CovariateEffect::CONTINUOUS.to_vec(),
                Modes::List(list) => list.clone(),
            };
            for e in effects {
                match e {
                    CovariateEffect::Cat2 => gap(
                        gaps,
                        "COVARIATE(..., cat2)".into(),
                        "`[covariate_model]` has one categorical form, `categorical(ref = …)` \
                         (MFL `cat`); there is no `cat2` spelling",
                    ),
                    CovariateEffect::Custom => gap(
                        gaps,
                        "COVARIATE(..., custom)".into(),
                        "a search cannot invent an expression; write the relation with the \
                         `expr(...)` escape hatch in the base model's `[covariate_model]` \
                         instead",
                    ),
                    _ => {}
                }
            }
            if *op == CovariateOp::Add {
                gap(
                    gaps,
                    "COVARIATE(..., +)".into(),
                    "`[covariate_model]` relations are multiplicative factors on a \
                     top-level product; an additive (`+`) effect has no spelling",
                );
            }
        }
        Feature::Allometry { .. } => {}
        Feature::DirectEffect(_) => gap(
            gaps,
            "DIRECTEFFECT(...)".into(),
            "there is no PD template family; PD structures are `[odes]` text only",
        ),
        Feature::EffectComp(_) => gap(
            gaps,
            "EFFECTCOMP(...)".into(),
            "there is no PD template family; PD structures are `[odes]` text only",
        ),
        Feature::IndirectEffect { .. } => gap(
            gaps,
            "INDIRECTEFFECT(...)".into(),
            "there is no PD template family; PD structures are `[odes]` text only",
        ),
        Feature::Metabolite(_) => gap(
            gaps,
            "METABOLITE(...)".into(),
            "there is no metabolite template family (structsearch is out of scope for \
             #1175 v1)",
        ),
        Feature::Iiv { effects, .. } => {
            for e in effects.expand() {
                if !iiv_effect_supported(e) {
                    gap(
                        gaps,
                        format!("IIV(..., {})", e.label()),
                        "`ferx-core::edit` writes an η as `exp` (`P = TVP * exp(ETA)`), \
                         `add` or `prop` only",
                    );
                }
            }
        }
        Feature::Iov { effects, .. } => {
            for e in effects.expand() {
                if e != VariabilityEffect::Exp {
                    gap(
                        gaps,
                        format!("IOV(..., {})", e.label()),
                        "`ferx-core::edit` writes a κ inside the exponential only \
                         (`P = TVP * exp(ETA + KAPPA)`), which is the form `[iov]` estimates",
                    );
                }
            }
        }
        // `COVARIANCE(IIV, …)` is `block_omega`, `COVARIANCE(IOV, …)` is
        // `block_kappa` (#1183); both levels are buildable.
        Feature::Covariance { .. } => {}
    }
}

/// The variability effects `ferx-core::edit::IivForm` can write.
pub fn iiv_effect_supported(effect: VariabilityEffect) -> bool {
    matches!(
        effect,
        VariabilityEffect::Exp | VariabilityEffect::Add | VariabilityEffect::Prop
    )
}

#[cfg(test)]
#[path = "coverage_tests.rs"]
mod tests;
