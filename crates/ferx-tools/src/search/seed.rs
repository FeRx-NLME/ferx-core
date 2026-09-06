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
/// [`MIN_SEED_VARIANCE`] so a parent whose η collapsed still yields a model
/// the engine will start.
///
/// Raising only the diagonal keeps a block positive-definite, and the floored
/// variance is the same no-variability model the parent found — spelled so
/// the child can move off it.
pub(crate) fn seed_from(model: &mut ModelText, fit: &FitResult) -> Result<(), String> {
    let floored = floor_variances(fit);
    model.apply(ModelEdit::SeedInits(floored.as_ref().unwrap_or(fit)))
}

/// `fit` with its ω diagonal floored, or `None` when nothing was below the
/// floor — the common case, and the one that costs no clone.
fn floor_variances(fit: &FitResult) -> Option<FitResult> {
    let below: Vec<usize> = (0..fit.omega.nrows())
        .filter(|k| {
            let v = fit.omega[(*k, *k)];
            v.is_finite() && v < MIN_SEED_VARIANCE
        })
        .collect();
    if below.is_empty() {
        return None;
    }
    let mut out = fit.clone();
    for k in below {
        out.omega[(k, k)] = MIN_SEED_VARIANCE;
    }
    Some(out)
}

#[cfg(test)]
#[path = "seed_tests.rs"]
mod tests;
