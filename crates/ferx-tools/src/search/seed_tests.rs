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

/// The #1256 vancomycin case: every declared variance is ordinary, but the
/// block's correlations leave `ETA_V2` with a Cholesky diagonal of 6.1e-6 —
/// the engine refuses to start a child there (`E_OMEGA_INIT_AT_RAIL`). The
/// seed nudges the block by `δ·I` until its factor is above the rail.
#[test]
fn seed_from_regularises_a_block_whose_cholesky_diagonal_is_at_the_rail() {
    // A 3×3 block whose third Schur complement is exactly the rail.
    let mut fit = fixture_fit();
    let names = fit.eta_names.clone();
    assert_eq!(names.len(), 3, "the warfarin fixture has three η");
    let rail: f64 = 6.144_212_353_328_21e-6;
    // ω = L Lᵀ with L = [[a,0,0],[b,c,0],[d,e,f]], f² = rail.
    let (a, b, c, d, e, f) = (0.6f64, 0.3f64, 0.1f64, 0.2f64, 0.9f64, rail.sqrt());
    let l = nalgebra::DMatrix::from_row_slice(3, 3, &[a, 0.0, 0.0, b, c, 0.0, d, e, f]);
    fit.omega = &l * l.transpose();
    assert!(
        (0..3).all(|k| fit.omega[(k, k)] > MIN_SEED_VARIANCE),
        "every declared variance is ordinary: {:?}",
        fit.omega
    );
    assert!(!super::cholesky_above_rail(&fit.omega));

    // A model that blocks the three η, so `SeedInits` writes the block.
    let src = std::fs::read_to_string(MODEL).unwrap();
    let src = src
        .replace("  omega ETA_CL ~ 0.09\n", "")
        .replace("  omega ETA_V  ~ 0.04\n", "")
        .replace(
            "  omega ETA_KA ~ 0.30\n",
            "  block_omega (ETA_CL, ETA_V, ETA_KA) = [0.09, 0.0, 0.04, 0.0, 0.0, 0.30]\n",
        );
    let mut model = ModelText::parse(&src).unwrap();
    seed_from(&mut model, &fit).expect("seeding a child");
    let line = model
        .block_lines("parameters")
        .into_iter()
        .find(|l| l.starts_with("block_omega"))
        .expect("the block is written");
    // Read the written triangle back and factor it: every Cholesky diagonal
    // is now above the floor, and the declared variances moved by one δ.
    let tri: Vec<f64> = line
        .split_once('[')
        .unwrap()
        .1
        .trim_end_matches(']')
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    assert_eq!(tri.len(), 6, "{line}");
    let mut written = nalgebra::DMatrix::zeros(3, 3);
    let mut i = 0;
    for r in 0..3 {
        for c in 0..=r {
            written[(r, c)] = tri[i];
            written[(c, r)] = tri[i];
            i += 1;
        }
    }
    assert!(super::cholesky_above_rail(&written), "{line}");
    for k in 0..3 {
        let moved = written[(k, k)] - fit.omega[(k, k)];
        assert!(
            (moved - MIN_SEED_VARIANCE).abs() < 1e-12,
            "diagonal {k} moved by {moved:e}, not one δ"
        );
    }
    // One nudge of δ lifts the collapsed Schur complement to at least δ.
    let l_new = nalgebra::Cholesky::new(written).unwrap();
    let l_new = l_new.l();
    assert!(l_new[(2, 2)] * l_new[(2, 2)] >= MIN_SEED_VARIANCE);

    // The mutation this pins: `SeedInits` alone writes the singular block.
    let mut verbatim = ModelText::parse(&src).unwrap();
    verbatim.apply(ModelEdit::SeedInits(&fit)).unwrap();
    let raw = verbatim
        .block_lines("parameters")
        .into_iter()
        .find(|l| l.starts_with("block_omega"))
        .unwrap();
    assert_ne!(raw, line);
}

/// A healthy block is left alone: no nudge, no clone.
#[test]
fn seed_from_leaves_a_well_conditioned_block_alone() {
    let fit = fixture_fit();
    assert!(super::cholesky_above_rail(&fit.omega));
    let blocks = super::free_blocks_of(&warfarin_text(), &fit.eta_names);
    assert!(blocks.is_empty(), "warfarin's ωs are diagonal");
    assert!(super::floor_variances(&fit, &[vec![0, 1, 2]]).is_none());
}

/// The review on #1265: the nudge is scoped to the failing block. A
/// near-singular `(ETA_CL, ETA_V)` block beside a healthy standalone
/// `ETA_KA` at 2e-5: the block is nudged, `ETA_KA` is written verbatim.
#[test]
fn seed_from_nudges_only_the_failing_block_not_a_healthy_standalone_omega() {
    let mut fit = fixture_fit();
    let names = fit.eta_names.clone();
    let cl = names.iter().position(|n| n == "ETA_CL").unwrap();
    let v = names.iter().position(|n| n == "ETA_V").unwrap();
    let ka = names.iter().position(|n| n == "ETA_KA").unwrap();
    // (CL, V) = L Lᵀ with the second Cholesky diagonal on the rail.
    let rail: f64 = 6.144_212_353_328_21e-6;
    let (a, b, c) = (0.5f64, 0.3f64, rail.sqrt());
    fit.omega[(cl, cl)] = a * a;
    fit.omega[(cl, v)] = a * b;
    fit.omega[(v, cl)] = a * b;
    fit.omega[(v, v)] = b * b + c * c;
    fit.omega[(ka, ka)] = 2e-5;
    fit.omega[(ka, cl)] = 0.0;
    fit.omega[(cl, ka)] = 0.0;
    fit.omega[(ka, v)] = 0.0;
    fit.omega[(v, ka)] = 0.0;

    let src = std::fs::read_to_string(MODEL).unwrap();
    let src = src.replace(
        "omega ETA_CL ~ 0.09\n  omega ETA_V  ~ 0.04",
        "block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04]",
    );
    let mut model = ModelText::parse(&src).unwrap();
    seed_from(&mut model, &fit).expect("seeding a child");
    let params = model.block_lines("parameters");
    assert!(
        params.contains(&"omega ETA_KA ~ 0.00002".to_string()),
        "the standalone ω is untouched: {params:?}"
    );
    let block = params
        .iter()
        .find(|l| l.starts_with("block_omega"))
        .unwrap();
    let tri: Vec<f64> = block
        .split_once('[')
        .unwrap()
        .1
        .trim_end_matches(']')
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    assert!(
        (tri[0] - (a * a + MIN_SEED_VARIANCE)).abs() < 1e-12,
        "{block}"
    );
    assert!(
        (tri[1] - a * b).abs() < 1e-12,
        "the covariance is verbatim: {block}"
    );
    assert!(
        (tri[2] - (b * b + c * c + MIN_SEED_VARIANCE)).abs() < 1e-12,
        "{block}"
    );
}

/// A `FIX`ed block is never nudged, and is never a reason to touch anything
/// else: the engine exempts it from the rail, and `SeedInits` leaves a `FIX`
/// line alone. The standalone ω beside it is seeded verbatim.
#[test]
fn seed_from_leaves_a_fixed_block_and_its_neighbours_alone() {
    let mut fit = fixture_fit();
    let names = fit.eta_names.clone();
    let cl = names.iter().position(|n| n == "ETA_CL").unwrap();
    let v = names.iter().position(|n| n == "ETA_V").unwrap();
    let ka = names.iter().position(|n| n == "ETA_KA").unwrap();
    fit.omega[(cl, cl)] = 0.25;
    fit.omega[(cl, v)] = 0.15;
    fit.omega[(v, cl)] = 0.15;
    fit.omega[(v, v)] = 0.09; // exactly singular: 0.15² = 0.25 · 0.09
    fit.omega[(ka, ka)] = 2e-5;
    let src = std::fs::read_to_string(MODEL).unwrap();
    let src = src.replace(
        "omega ETA_CL ~ 0.09\n  omega ETA_V  ~ 0.04",
        "block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04] FIX",
    );
    let mut model = ModelText::parse(&src).unwrap();
    seed_from(&mut model, &fit).expect("seeding a child");
    let params = model.block_lines("parameters");
    assert!(
        params.contains(&"block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04] FIX".to_string()),
        "{params:?}"
    );
    assert!(
        params.contains(&"omega ETA_KA ~ 0.00002".to_string()),
        "{params:?}"
    );
    assert!(super::free_blocks_of(&model, &names).is_empty());
}

/// Past `MAX_NUDGES` the block goes through verbatim — not sixteen nudges
/// that still do not start. An indefinite block (covariance far larger than
/// its variances) never becomes positive-definite by `16·δ`.
#[test]
fn seed_from_hands_an_unrepairable_block_through_verbatim() {
    let mut fit = fixture_fit();
    let names = fit.eta_names.clone();
    let cl = names.iter().position(|n| n == "ETA_CL").unwrap();
    let v = names.iter().position(|n| n == "ETA_V").unwrap();
    fit.omega[(cl, cl)] = 1e-5;
    fit.omega[(v, v)] = 1e-5;
    fit.omega[(cl, v)] = 1e-3;
    fit.omega[(v, cl)] = 1e-3;
    let src = std::fs::read_to_string(MODEL).unwrap();
    let src = src.replace(
        "omega ETA_CL ~ 0.09\n  omega ETA_V  ~ 0.04",
        "block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04]",
    );
    let blocks = super::free_blocks_of(&ModelText::parse(&src).unwrap(), &names);
    assert_eq!(blocks, vec![vec![cl, v]]);
    let floored = super::floor_variances(&fit, &blocks);
    // Nothing else was below the floor, and the block was given back as it
    // was: no change at all.
    assert!(floored.is_none(), "{:?}", floored.map(|f| f.omega));

    let mut model = ModelText::parse(&src).unwrap();
    seed_from(&mut model, &fit).expect("seeding a child");
    let mut verbatim = ModelText::parse(&src).unwrap();
    verbatim.apply(ModelEdit::SeedInits(&fit)).unwrap();
    assert_eq!(
        model.block_lines("parameters"),
        verbatim.block_lines("parameters")
    );
}
