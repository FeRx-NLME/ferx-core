//! Tier-1 unit tests for the combined-error additive-init scale warning
//! (`W_ADDITIVE_INIT_SCALE`). The module is a child of `validation`, so
//! `super::additive_init_scale_check` resolves the private decision helper
//! directly — no full `CompiledModel`/`Population` construction needed.
//!
//! Regression anchor: the cyclophosphamide parent→metabolite fit, where an
//! additive variance seeded at 0.5 (SD ≈ 0.707) against metabolite
//! concentrations in the hundreds/thousands ng/mL traps both ferx and NONMEM in
//! the additive-collapse local minimum. This warning fires on exactly that
//! start.

use super::additive_init_scale_check;

/// A negligible additive SD (well under 1% of the data scale) fires the warning
/// and the message carries the diagnostic code, the endpoint scope, and a
/// larger suggested start.
#[test]
fn negligible_additive_init_warns() {
    // Metabolite-like concentrations; median = 200.
    let mut mags = vec![50.0, 100.0, 200.0, 400.0, 1000.0];
    // sqrt(0.5) ≈ 0.707 — the cyclophosphamide trap start.
    let d = additive_init_scale_check("CMT=2", "ADD_M1", 0.707, &mut mags)
        .expect("negligible additive init must warn");
    assert_eq!(d.code, "W_ADDITIVE_INIT_SCALE");
    assert!(d.message.contains("CMT=2"));
    assert!(d.message.contains("ADD_M1"));
    // Suggested start is 10% of the median (200) = 20.
    assert!(
        d.message.contains("20.0000"),
        "message should suggest ~10% of data scale, got: {}",
        d.message
    );
}

/// An additive SD at exactly the 1% threshold is not negligible (strict `<`).
#[test]
fn additive_init_at_threshold_is_silent() {
    let mut mags = vec![100.0, 200.0, 300.0]; // median 200 → threshold 2.0
    assert!(additive_init_scale_check("CMT=2", "ADD_M1", 2.0, &mut mags).is_none());
}

/// A sensibly-scaled additive floor (a few % of the data) does not warn — the
/// threshold is conservative by design, so well-specified combined models stay
/// quiet.
#[test]
fn reasonable_additive_init_is_silent() {
    let mut mags = vec![50.0, 100.0, 200.0, 400.0, 1000.0]; // median 200
                                                            // Pumas optimum additive SD sqrt(1878) ≈ 43.3 — far above 1% of 200.
    assert!(additive_init_scale_check("CMT=2", "ADD_M1", 43.3, &mut mags).is_none());
    // Even a modest 5-of-200 (2.5%) floor stays silent.
    assert!(additive_init_scale_check("CMT=2", "ADD_M1", 5.0, &mut mags).is_none());
}

/// No observations (empty magnitudes) → nothing to compare against → silent.
#[test]
fn empty_observations_is_silent() {
    let mut mags: Vec<f64> = vec![];
    assert!(additive_init_scale_check("CMT=2", "ADD_M1", 0.001, &mut mags).is_none());
}

/// An all-zero data scale (median |DV| = 0) is silent — dividing the threshold
/// by it is meaningless and there is no scale to anchor a suggestion.
#[test]
fn zero_data_scale_is_silent() {
    let mut mags = vec![0.0, 0.0, 0.0];
    assert!(additive_init_scale_check("CMT=2", "ADD_M1", 0.001, &mut mags).is_none());
}

/// Median is computed correctly for an even count (mean of the two middle
/// order statistics) and the comparison uses it. Even-length median of
/// [10,20,30,40] is 25 → threshold 0.25; an init of 0.2 warns, 0.3 does not.
#[test]
fn even_length_median_threshold() {
    let mut a = vec![40.0, 10.0, 30.0, 20.0];
    assert!(additive_init_scale_check("CMT=1", "ADD", 0.2, &mut a).is_some());
    let mut b = vec![40.0, 10.0, 30.0, 20.0];
    assert!(additive_init_scale_check("CMT=1", "ADD", 0.3, &mut b).is_none());
}
