//! SAEM option defaults the parser resolves. Below the estimators (#3041).

/// Sanitise the `scale_deadband` fit option at its point of use.
///
/// The parser validates what a user *types*, but `FitOptions` is public — the
/// ferx-r glue, `ferx-tools`, a test, any embedder can build one directly and
/// never pass through that validator (#1479 review). The failure mode is the
/// worst one this option has: `Some((0.0, 1.0))` puts every attainable
/// acceptance rate inside the band, so [`inside_scale_deadband`] answers `true`
/// on every iteration and **both scale controllers freeze for the whole fit**
/// — silently, and with the Robbins-Monro rule now on by default.
///
/// Returns `Some(substituted)` when the band had to be changed, so the caller
/// can warn, and `None` when it was already in contract — the same shape as
/// [`sanitize_mstep_damping`]. The rules, applied in this order:
///
/// 1. a non-finite end drops the band (`NaN` already compares `false`
///    everywhere, but `(-inf, inf)` would freeze the fit);
/// 2. ends are clamped into `[0, 1]`, which is where an acceptance rate lives,
///    so `(-0.1, 0.5)` means what it looks like rather than being refused;
/// 3. an empty or reversed band (`lo >= hi` after clamping) drops to `None`,
///    matching the parser's `lo == hi` normalisation — an interval containing
///    nothing cannot skip anything;
/// 4. a band covering all of `[0, 1]` drops to `None`, which is the *rule
///    without the band* rather than the frozen fit it would otherwise be.
///
/// Rule 4 is the one place the two front ends differ, and deliberately: a model
/// file saying `scale_deadband = 0,1` is **rejected** so the user fixes the
/// typo, while a Rust caller gets the band dropped and a warning. Neither
/// freezes, which is the property that matters; erroring out of `run_saem` on a
/// field an embedder may have defaulted badly would turn a recoverable
/// misconfiguration into a failed fit.
pub(crate) fn sanitize_scale_deadband(band: Option<(f64, f64)>) -> Option<Option<(f64, f64)>> {
    let (lo, hi) = band?;
    if !lo.is_finite() || !hi.is_finite() {
        return Some(None);
    }
    let (clo, chi) = (lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0));
    if clo >= chi || (clo <= 0.0 && chi >= 1.0) {
        return Some(None);
    }
    if clo == lo && chi == hi {
        None
    } else {
        Some(Some((clo, chi)))
    }
}

/// Value of [`FitOptions::saem_n_mh_steps`] that selects
/// [`auto_n_mh_steps`] — the shipped default, written `n_mh_steps = auto` in a
/// model file.
pub(crate) const SAEM_N_MH_STEPS_AUTO: usize = 0;
