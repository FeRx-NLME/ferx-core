//! Unit tests for [`seed_from`](super::seed_from).

use ferx_core::edit::{ModelEdit, ModelText};

use super::{seed_from, MIN_SEED_VARIANCE};
use crate::search::test_support::{fixture_fit, MODEL};

fn warfarin_text() -> ModelText {
    ModelText::parse(&std::fs::read_to_string(MODEL).expect("the warfarin model file"))
        .expect("the warfarin model parses")
}

/// The floor exists for the #1181 case: a parent whose η collapsed onto the
/// optimizer's rail (`e⁻¹² ≈ 6.14e-6`) hands its child a start the engine
/// refuses, so the child cannot be fitted at all.
#[test]
fn seed_from_floors_a_collapsed_variance_at_the_smallest_startable_one() {
    let mut fit = fixture_fit();
    let k = fit
        .eta_names
        .iter()
        .position(|n| n == "ETA_KA")
        .expect("the warfarin model has ETA_KA");
    fit.omega[(k, k)] = 6.144_212_353_328_21e-6;

    let mut model = warfarin_text();
    seed_from(&mut model, &fit).expect("seeding a child");
    let params = model.block_lines("parameters");
    assert!(
        params.contains(&"omega ETA_KA ~ 0.00001".to_string()),
        "the collapsed variance is raised to the floor: {params:?}"
    );

    // The mutation this pins: without the floor the child is written at the
    // rail, which is exactly what `SeedInits` on its own does.
    let mut verbatim = warfarin_text();
    verbatim
        .apply(ModelEdit::SeedInits(&fit))
        .expect("the unfloored edit");
    let raw = verbatim.block_lines("parameters");
    assert!(
        raw.contains(&"omega ETA_KA ~ 0.00000614421235332821".to_string()),
        "`SeedInits` alone stays faithful to the fit: {raw:?}"
    );
}

/// A healthy fit is seeded verbatim — the floor is not a rounding of every
/// small variance, and the common case clones nothing.
#[test]
fn seed_from_leaves_a_healthy_variance_alone() {
    let fit = fixture_fit();
    assert!(
        (0..fit.omega.nrows()).all(|k| fit.omega[(k, k)] > MIN_SEED_VARIANCE),
        "the fixture's ωs are all above the floor, which is what makes this a control"
    );
    let mut floored = warfarin_text();
    seed_from(&mut floored, &fit).expect("seeding a child");
    let mut verbatim = warfarin_text();
    verbatim
        .apply(ModelEdit::SeedInits(&fit))
        .expect("the unfloored edit");
    assert_eq!(
        floored.block_lines("parameters"),
        verbatim.block_lines("parameters"),
        "nothing below the floor: the two edits agree line for line"
    );
}

/// Only the diagonal is raised: an off-diagonal covariance is a correlation,
/// and raising it would change the block rather than keep it startable.
#[test]
fn seed_from_raises_the_diagonal_only() {
    let mut fit = fixture_fit();
    let n = fit.omega.nrows();
    assert!(n >= 2, "the fixture needs two ηs for an off-diagonal");
    for k in 0..n {
        fit.omega[(k, k)] = 1e-9;
    }
    fit.omega[(1, 0)] = 1e-7;
    fit.omega[(0, 1)] = 1e-7;

    let src = std::fs::read_to_string(MODEL).expect("the warfarin model file");
    let src = src.replace(
        "omega ETA_CL ~ 0.09\n  omega ETA_V  ~ 0.04",
        "block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04]",
    );
    let mut model = ModelText::parse(&src).expect("the block model parses");
    seed_from(&mut model, &fit).expect("seeding a child");
    let params = model.block_lines("parameters");
    assert!(
        params.contains(&"block_omega (ETA_CL, ETA_V) = [0.00001, 0.0000001, 0.00001]".to_string()),
        "diagonal floored, off-diagonal verbatim: {params:?}"
    );
}
