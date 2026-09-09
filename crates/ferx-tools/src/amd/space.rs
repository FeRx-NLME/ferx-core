//! Splitting one AMD search space into the subspace each tool will accept
//! (#1184).
//!
//! Every search tool of the epic *refuses* a statement that is not its own —
//! `modelsearch` on a `COVARIATE`, `covsearch` on an `ABSORPTION`, `iivsearch`
//! on an `IOV`. That is deliberate (a file meant for another tool must not be
//! run half-silently), and it is exactly what a pipeline driven by **one**
//! `[space] mfl` runs into: the AMD space is the union of all of them.
//!
//! So AMD partitions the space by statement kind before it hands a config to a
//! tool, and each step sees only the statements it can read. Three properties
//! make the partition trustworthy rather than convenient:
//!
//! * **Nothing is dropped.** Every feature lands in exactly one step's
//!   subspace, except `COVARIANCE(*, …)` / `COVARIANCE([IIV,IOV], …)`, which
//!   names two levels and is therefore narrowed to `COVARIANCE(IIV, …)` for
//!   `iivsearch` and `COVARIANCE(IOV, …)` for `iovsearch` — the two halves of
//!   what it says. A statement no step can read is an **error naming it**, not
//!   a narrowed search.
//! * **`LET` goes to everyone.** A `LET` defines a symbol the statements refer
//!   to (`@MYPARAMS`), so it is copied into every subspace; an unused
//!   definition costs nothing, a missing one fails resolution.
//! * **The order is the source's.** Statements keep their relative order inside
//!   a subspace, so a rendered subspace re-parses to the same features and two
//!   runs of the same file build the same candidates.

use super::Step;
use crate::search::mfl::{Modes, VariabilityLevel};
use crate::search::{Feature, Mfl, Statement};

/// The step that owns one feature, or `None` when two do.
fn owner(feature: &Feature) -> Result<Option<Step>, String> {
    Ok(Some(match feature {
        Feature::Absorption(_)
        | Feature::Elimination(_)
        | Feature::Peripherals { .. }
        | Feature::Transits { .. }
        | Feature::Lagtime(_) => Step::Structural,
        Feature::Covariate { .. } => Step::Covariates,
        Feature::Allometry { .. } => Step::Allometry,
        Feature::Iiv { .. } => Step::Iivsearch,
        Feature::Iov { .. } => Step::Iovsearch,
        // Both levels at once: split below rather than assigned here.
        Feature::Covariance { .. } => return Ok(None),
        other => {
            return Err(format!(
                "[space] mfl: `{}` is a PD / metabolite statement, which the AMD pipeline has no \
                 step for — its structural step is `modelsearch` (PK), and `structsearch` is not \
                 implemented yet (#1175). Remove the statement or run the tool it belongs to on \
                 its own",
                other.keyword()
            ))
        }
    }))
}

/// `feature` restricted to `level`, when it says anything about that level.
///
/// Only `COVARIANCE` reaches this. `COVARIANCE(IIV, …)` narrowed to `IIV` is
/// itself; narrowed to `IOV` is nothing. `COVARIANCE(*, …)` narrows to a
/// single-level statement on either side, which is what makes the two halves
/// add up to the whole: `iivsearch` reads the η blocks and `iovsearch` the κ,
/// and neither is handed a statement mentioning the other's level.
fn narrowed(feature: &Feature, level: VariabilityLevel) -> Option<Feature> {
    let Feature::Covariance {
        optional,
        level: levels,
        parameters,
    } = feature
    else {
        return None;
    };
    levels
        .expand()
        .contains(&level)
        .then(|| Feature::Covariance {
            optional: *optional,
            level: Modes::List(vec![level]),
            parameters: parameters.clone(),
        })
}

/// The subspace `step` is handed: its own statements, every `LET`, and nothing
/// else.
///
/// Returns an empty `Mfl` (no features) when the space says nothing about this
/// step — which the caller reads as "skip", not as "search everything".
pub(crate) fn subspace(mfl: &Mfl, step: Step) -> Result<Mfl, String> {
    let mut statements = Vec::new();
    for statement in &mfl.statements {
        match statement {
            Statement::Let { .. } => statements.push(statement.clone()),
            Statement::Feature(feature) => {
                let keep = match owner(feature)? {
                    Some(owner) => (owner == step).then(|| feature.clone()),
                    None => match step {
                        Step::Iivsearch => narrowed(feature, VariabilityLevel::Iiv),
                        Step::Iovsearch => narrowed(feature, VariabilityLevel::Iov),
                        _ => None,
                    },
                };
                if let Some(feature) = keep {
                    statements.push(Statement::Feature(feature));
                }
            }
        }
    }
    // A subspace of `LET`s only is empty: a definition nothing refers to is
    // not a search.
    if !statements
        .iter()
        .any(|s| matches!(s, Statement::Feature(_)))
    {
        statements.clear();
    }
    Ok(Mfl { statements })
}

/// Every statement's owner, checked, without building the subspaces — so a
/// space AMD cannot route is refused before the base model is read.
pub(crate) fn check(mfl: &Mfl) -> Result<(), String> {
    for feature in mfl.features() {
        owner(feature)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "space_tests.rs"]
mod tests;
