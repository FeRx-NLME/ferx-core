//! Seeding a search child from its parent's fit.
//!
//! [`ModelEdit::SeedInits`] reproduces the fit it is handed — that is what
//! makes a written `final.ferx` re-evaluate to the OFV beside it in
//! `final-fit.yaml`. A *search child* needs one thing more: a start it can
//! actually leave.
//!
//! The optimizer packs a diagonal ω as `ln(L)` with a lower bound of −6
//! (`estimation/parameterization.rs`), so any variance at or below
//! `e⁻¹² ≈ 6.14e-6` starts *on* the rail, and the engine refuses such a start
//! rather than clamp it silently (`api/validation.rs` names `1e-5` as the
//! smallest it accepts). A parent whose η collapsed there — measured on
//! #1181, a lag time whose ω fell to 6.1e-6 — would otherwise hand every
//! child a model that cannot start at all.
//!
//! So the floor is applied here, to the **fit the child is seeded from**,
//! and not inside the shared edit: a search *reporting* its winner writes the
//! estimates verbatim, and only a search *deriving* a child raises a
//! collapsed variance to the smallest startable one. Both searches of the
//! epic route through [`seed_from`], so the policy has one implementation.

use ferx_core::edit::{ModelEdit, ModelText};
use ferx_core::FitResult;

/// The smallest variance a seeded child starts an η at. See the module docs.
pub(crate) const MIN_SEED_VARIANCE: f64 = 1e-5;

/// Carry `fit`'s estimates into `model` as a **child's** starting values:
/// [`ModelEdit::SeedInits`], with every ω diagonal raised to
/// [`MIN_SEED_VARIANCE`] and every ω block regularised so its Cholesky
/// diagonal is above the rail too — a parent whose η collapsed, on its own
/// or through a block's correlations, still yields a model the engine will
/// start.
///
/// The floored variance is the same no-variability model the parent found —
/// spelled so the child can move off it.
pub(crate) fn seed_from(model: &mut ModelText, fit: &FitResult) -> Result<(), String> {
    let floored = floor_variances(fit);
    model.apply(ModelEdit::SeedInits(floored.as_ref().unwrap_or(fit)))
}

/// How many times a near-singular ω may be nudged by `MIN_SEED_VARIANCE·I`
/// before the seed gives up and hands the parent's ω through verbatim (so
/// the engine's own message names the block).
const MAX_NUDGES: usize = 16;

/// `fit` with its ω made startable, or `None` when it already was — the
/// common case, and the one that costs no clone.
///
/// Two things put an ω on the rail. A collapsed **diagonal** (`ω_kk ≤ e⁻¹²`)
/// is raised to the floor. A **block** whose correlations make it
/// near-singular — the #1256 vancomycin case, where every declared variance
/// is ordinary but the Cholesky diagonal for `ETA_V2` is what is left of its
/// variance after the covariances, 6.1e-6 — is nudged by `δ·I`, δ the same
/// floor: for a positive-semidefinite ω every Schur complement of `ω + δI`
/// is at least δ, so one nudge lifts every Cholesky diagonal above the rail
/// while moving the declared variances by a part in 10⁵ of the floor. A
/// fitted ω that is slightly *indefinite* in the last digits may need more
/// than one; the loop is bounded and, past it, the seed is verbatim.
fn floor_variances(fit: &FitResult) -> Option<FitResult> {
    let n = fit.omega.nrows();
    let mut omega = fit.omega.clone();
    let mut changed = false;
    for k in 0..n {
        let v = omega[(k, k)];
        if v.is_finite() && v < MIN_SEED_VARIANCE {
            omega[(k, k)] = MIN_SEED_VARIANCE;
            changed = true;
        }
    }
    // The block check is on the whole matrix: outside a block ω is diagonal,
    // so its Cholesky diagonals are the per-block ones and the floored
    // variances above.
    let mut nudges = 0;
    while nudges < MAX_NUDGES && !cholesky_above_rail(&omega) {
        for k in 0..n {
            omega[(k, k)] += MIN_SEED_VARIANCE;
        }
        changed = true;
        nudges += 1;
    }
    if !changed {
        return None;
    }
    let mut out = fit.clone();
    out.omega = omega;
    Some(out)
}

/// The engine's rail on a packed Cholesky diagonal, as a variance:
/// `ln(L) = −6`, i.e. `L² = e⁻¹²`. A start at or below it is refused
/// (`api/validation.rs`).
const RAIL_VARIANCE: f64 = 6.144_212_353_328_21e-6;

/// The smallest Cholesky diagonal (squared) the block check accepts: the
/// rail with headroom. Deliberately *below* [`MIN_SEED_VARIANCE`] — a block
/// whose diagonals were just floored to 1e-5 and whose covariance is small
/// has a Cholesky diagonal a hair under 1e-5, and is startable; nudging it
/// again would move the declared variances for nothing.
const MIN_CHOLESKY_VARIANCE: f64 = 8e-6;

/// Whether `omega` has a Cholesky factor whose every diagonal entry, squared,
/// is above the rail with headroom ([`MIN_CHOLESKY_VARIANCE`]). A matrix
/// that is not positive-definite has no factor and fails too.
fn cholesky_above_rail(omega: &nalgebra::DMatrix<f64>) -> bool {
    debug_assert!(MIN_CHOLESKY_VARIANCE > RAIL_VARIANCE);
    if omega.iter().any(|v| !v.is_finite()) {
        // Nothing to regularise: `SeedInits` skips a non-finite entry.
        return true;
    }
    match nalgebra::Cholesky::new(omega.clone()) {
        Some(chol) => {
            let l = chol.l();
            (0..omega.nrows()).all(|k| l[(k, k)] * l[(k, k)] >= MIN_CHOLESKY_VARIANCE)
        }
        None => false,
    }
}

#[cfg(test)]
#[path = "seed_tests.rs"]
mod tests;
