//! `correlation cs|ar1(RHO) by COL` — the serial residual blocks of #1081.
//!
//! One subject, two arms of three visits each, the rows **interleaved and out of
//! time order** so that block membership and the AR(1) time order are both read
//! from the data rather than from row adjacency. Combined error with a `weight`
//! (the #484 multiplier on the additive slot) so that the off-diagonal carries
//! both the `f_j f_k σ_p²` and the `w_j w_k σ_a²` terms.

use super::*;
use crate::types::{ResidualSerialKind, Subject};
use approx::assert_relative_eq;
use std::collections::HashMap;

/// Rows: (arm, time). Arm 1 visits at 0, 2, 12; arm 2 at 0, 1, 5.
const ROWS: [(u32, f64); 6] = [(1, 12.0), (2, 0.0), (1, 0.0), (2, 5.0), (1, 2.0), (2, 1.0)];
const IPREDS: [f64; 6] = [30.0, 45.0, 50.0, 20.0, 38.0, 41.0];
const WEIGHTS: [f64; 6] = [1.3, 0.7, 2.0, 0.9, 1.6, 1.1];
/// Combined error: (proportional SD, additive SD).
const SIGMA: [f64; 2] = [0.2, 1.5];

fn rgroup() -> Vec<u32> {
    ROWS.iter().map(|r| r.0).collect()
}
fn times() -> Vec<f64> {
    ROWS.iter().map(|r| r.1).collect()
}
/// `weight = W` is the #484 multiplier on the additive slot (slot 1).
fn mult() -> Vec<Vec<f64>> {
    WEIGHTS.iter().map(|&w| vec![1.0, w]).collect()
}
fn cov(kind: ResidualSerialKind, rho: f64) -> ResidualCov<'static> {
    ResidualCov {
        pairs: &[],
        serial: Some(SerialCov { kind, rho }),
    }
}
fn spec() -> ErrorSpec {
    ErrorSpec::Single(crate::types::ErrorModel::Combined)
}

/// Position of each row within its arm in time order — computed here by hand,
/// not by the code under test.
fn positions() -> Vec<i32> {
    // Arm 1: t 0 (row 2), 2 (row 4), 12 (row 0). Arm 2: t 0 (row 1), 1 (row 5), 5 (row 3).
    vec![2, 0, 0, 2, 1, 1]
}

fn closed_form_c(kind: ResidualSerialKind, rho: f64, j: usize, k: usize) -> f64 {
    if ROWS[j].0 != ROWS[k].0 {
        return 0.0;
    }
    if j == k {
        return 1.0;
    }
    match kind {
        ResidualSerialKind::Cs => rho,
        ResidualSerialKind::Ar1 => rho.powi((positions()[j] - positions()[k]).abs()),
    }
}

fn scaled_r(kind: ResidualSerialKind, rho: f64) -> DMatrix<f64> {
    compute_r_matrix_with_correlations_scaled_cov(
        &spec(),
        &IPREDS,
        &[1; 6],
        &times(),
        &[],
        &[],
        &[],
        &rgroup(),
        &SIGMA,
        cov(kind, rho),
        &mult(),
    )
}

/// T1: every entry of `R` equals `C_jk(ρ) · (f_j f_k σ_p² + w_j w_k σ_a²)`, for both
/// structures and both signs of ρ. Dies on: an off-diagonal without the weight
/// (unscaled loadings), a block spanning the two arms, AR(1) lags taken in row
/// order instead of time order, CS read as AR(1) or the reverse.
#[test]
fn serial_r_matches_the_closed_form() {
    for kind in [ResidualSerialKind::Cs, ResidualSerialKind::Ar1] {
        for rho in [0.6, -0.3] {
            let r = scaled_r(kind, rho);
            for j in 0..6 {
                for k in 0..6 {
                    let (fj, fk) = (IPREDS[j], IPREDS[k]);
                    let (wj, wk) = (WEIGHTS[j], WEIGHTS[k]);
                    let base = fj * fk * SIGMA[0] * SIGMA[0] + wj * wk * SIGMA[1] * SIGMA[1];
                    let want = closed_form_c(kind, rho, j, k) * base;
                    assert_relative_eq!(r[(j, k)], want, max_relative = 1e-13, epsilon = 1e-13);
                }
            }
        }
    }
    // The fixture really does straddle: rows 0 and 2 are one arm two lags apart
    // in time but two rows apart in file order too — so pin a pair where file
    // order and time order disagree (rows 0 and 4: adjacent in time, four rows apart).
    assert_eq!((positions()[0] - positions()[4]).abs(), 1);
    let r = scaled_r(ResidualSerialKind::Ar1, 0.6);
    let base04 = IPREDS[0] * IPREDS[4] * 0.04 + WEIGHTS[0] * WEIGHTS[4] * 2.25;
    assert_relative_eq!(r[(0, 4)], 0.6 * base04, max_relative = 1e-13);
}

/// A 2×2 by hand: additive SD 1, weights (1, 2), ρ = 0.5 ⇒ `R = [[1, 1], [1, 4]]`.
#[test]
fn serial_r_two_by_two_by_hand() {
    let r = compute_r_matrix_with_correlations_scaled_cov(
        &ErrorSpec::Single(crate::types::ErrorModel::Additive),
        &[3.0, 7.0],
        &[1, 1],
        &[0.0, 1.0],
        &[],
        &[],
        &[],
        &[1, 1],
        &[1.0],
        cov(ResidualSerialKind::Cs, 0.5),
        &[vec![1.0], vec![2.0]],
    );
    assert_eq!(r, DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 4.0]));
}

/// T2: at ρ = 0 the serial fill writes nothing — `R` is bit-identical to the
/// model without a `correlation` line, on the bare and the scaled builder. Dies on
/// a fill that writes `ρ⁰ = 1` (an AR(1) lag of 0 for every pair) or that touches
/// the diagonal.
#[test]
fn serial_r_at_rho_zero_is_the_independent_r_bit_for_bit() {
    let none = ResidualCov::block_sigma(&[]);
    for kind in [ResidualSerialKind::Cs, ResidualSerialKind::Ar1] {
        let scaled = |c| {
            compute_r_matrix_with_correlations_scaled_cov(
                &spec(),
                &IPREDS,
                &[1; 6],
                &times(),
                &[],
                &[],
                &[],
                &rgroup(),
                &SIGMA,
                c,
                &mult(),
            )
        };
        let bare = |c| {
            compute_r_matrix_with_correlations_cov(
                &spec(),
                &IPREDS,
                &[1; 6],
                &times(),
                &[],
                &[],
                &[],
                &rgroup(),
                &SIGMA,
                c,
            )
        };
        let bits = |m: DMatrix<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(scaled(cov(kind, 0.0))),
            bits(scaled(none)),
            "{kind:?} scaled"
        );
        assert_eq!(
            bits(bare(cov(kind, 0.0))),
            bits(bare(none)),
            "{kind:?} bare"
        );
        // and a live ρ does move it, so the comparison above is not vacuous
        assert_ne!(bits(bare(cov(kind, 0.5))), bits(bare(none)));
    }
}

/// T3: an additive endpoint with `weight` all ones (the scaled builder) and the
/// same endpoint with no weight (the bare builder) give the same `R` bit for bit.
/// Additive only: the bare and scaled diagonals associate an `f`-dependent term
/// differently (`variance_at` vs `variance_at_scaled`), which an additive slot
/// does not have.
#[test]
fn unit_weight_is_no_weight_on_an_additive_endpoint() {
    let spec = ErrorSpec::Single(crate::types::ErrorModel::Additive);
    let c = cov(ResidualSerialKind::Ar1, 0.45);
    let bare = compute_r_matrix_with_correlations_cov(
        &spec,
        &IPREDS,
        &[1; 6],
        &times(),
        &[],
        &[],
        &[],
        &rgroup(),
        &[1.5],
        c,
    );
    let ones = vec![vec![1.0]; 6];
    let scaled = compute_r_matrix_with_correlations_scaled_cov(
        &spec,
        &IPREDS,
        &[1; 6],
        &times(),
        &[],
        &[],
        &[],
        &rgroup(),
        &[1.5],
        c,
        &ones,
    );
    let bits = |m: &DMatrix<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&bare), bits(&scaled));
    assert!(bare[(0, 2)] != 0.0, "the fixture couples rows");
}

/// T5: `∂R/∂f` and `∂²R/∂f²` against central differences of the `R` builder,
/// bare and scaled, both structures. Combined error, so the proportional slot's
/// slope loading is live — an additive-only fixture has `∂R/∂f ≡ 0` off the
/// diagonal and could not see a missing serial term.
#[test]
fn serial_dr_and_d2r_match_finite_differences() {
    let n = 6;
    for kind in [ResidualSerialKind::Cs, ResidualSerialKind::Ar1] {
        for scaled in [false, true] {
            let m = mult();
            let mopt = scaled.then_some(m.as_slice());
            let c = cov(kind, 0.55);
            let r_at = |f: &[f64]| match mopt {
                Some(mm) => compute_r_matrix_with_correlations_scaled_cov(
                    &spec(),
                    f,
                    &[1; 6],
                    &times(),
                    &[],
                    &[],
                    &[],
                    &rgroup(),
                    &SIGMA,
                    c,
                    mm,
                ),
                None => compute_r_matrix_with_correlations_cov(
                    &spec(),
                    f,
                    &[1; 6],
                    &times(),
                    &[],
                    &[],
                    &[],
                    &rgroup(),
                    &SIGMA,
                    c,
                ),
            };
            let dr_at = |f: &[f64]| {
                compute_dr_df_matrices_cov(
                    &spec(),
                    f,
                    &[1; 6],
                    &times(),
                    &[],
                    &[],
                    &[],
                    &rgroup(),
                    &SIGMA,
                    c,
                    mopt,
                )
            };
            let dr = dr_at(&IPREDS);
            let d2r = compute_d2r_df2_matrices_cov(
                &spec(),
                &IPREDS,
                &[1; 6],
                &times(),
                &[],
                &[],
                &[],
                &rgroup(),
                &SIGMA,
                c,
                mopt,
            );
            let h = 1e-4;
            let mut off_diagonal_seen = false;
            for a in 0..n {
                let mut fp = IPREDS.to_vec();
                let mut fm = IPREDS.to_vec();
                fp[a] += h;
                fm[a] -= h;
                let fd = (r_at(&fp) - r_at(&fm)) / (2.0 * h);
                let (drp, drm) = (dr_at(&fp), dr_at(&fm));
                for p in 0..n {
                    for q in 0..n {
                        assert_relative_eq!(
                            dr[a][(p, q)],
                            fd[(p, q)],
                            epsilon = 1e-6,
                            max_relative = 1e-6
                        );
                        if p != q && fd[(p, q)].abs() > 1e-3 {
                            off_diagonal_seen = true;
                        }
                    }
                }
                for b in 0..n {
                    let fd2 = (&drp[b] - &drm[b]) / (2.0 * h);
                    for p in 0..n {
                        for q in 0..n {
                            assert_relative_eq!(
                                d2r[b][a][(p, q)],
                                fd2[(p, q)],
                                epsilon = 1e-6,
                                max_relative = 1e-6
                            );
                        }
                    }
                }
            }
            assert!(
                off_diagonal_seen,
                "{kind:?}: the fixture has a live ∂R_jk/∂f"
            );
        }
    }
}

/// `power(SIGMA, P)` (#1182): the loading `|f|^P` is not affine, so the serial
/// term has an own-row curvature `∂²R_jk/∂f_j² = C_jk · c_j'' c_k σ²` that the
/// affine loadings never produce. The mult row `[1, P]` carries the exponent.
#[test]
fn serial_d2r_carries_the_power_curvature() {
    let spec = ErrorSpec::Single(crate::types::ErrorModel::Proportional);
    let m: Vec<Vec<f64>> = (0..6).map(|_| vec![1.0, 1.4]).collect();
    let c = cov(ResidualSerialKind::Cs, 0.5);
    let dr_at = |f: &[f64]| {
        compute_dr_df_matrices_cov(
            &spec,
            f,
            &[1; 6],
            &times(),
            &[],
            &[],
            &[],
            &rgroup(),
            &[0.3],
            c,
            Some(&m),
        )
    };
    let d2r = compute_d2r_df2_matrices_cov(
        &spec,
        &IPREDS,
        &[1; 6],
        &times(),
        &[],
        &[],
        &[],
        &rgroup(),
        &[0.3],
        c,
        Some(&m),
    );
    let h = 1e-4;
    for a in 0..6 {
        let mut fp = IPREDS.to_vec();
        let mut fm = IPREDS.to_vec();
        fp[a] += h;
        fm[a] -= h;
        let fd2 = (&dr_at(&fp)[a] - &dr_at(&fm)[a]) / (2.0 * h);
        for p in 0..6 {
            for q in 0..6 {
                assert_relative_eq!(
                    d2r[a][a][(p, q)],
                    fd2[(p, q)],
                    epsilon = 1e-6,
                    max_relative = 1e-6
                );
            }
        }
    }
    // Row 0 and row 2 share arm 1: the own-row curvature is live off the diagonal.
    assert!(d2r[0][0][(0, 2)].abs() > 1e-6);
}

/// A population whose rows were never given their blocks has a NaN `R` (and NaN
/// derivatives), which fails the factorisation, rather than an independent `R`.
#[test]
fn unassigned_blocks_give_a_nan_r_not_an_independent_one() {
    let c = cov(ResidualSerialKind::Cs, 0.5);
    let r = compute_r_matrix_with_correlations_cov(
        &spec(),
        &IPREDS,
        &[1; 6],
        &times(),
        &[],
        &[],
        &[],
        &[],
        &SIGMA,
        c,
    );
    assert!(r.iter().all(|v| v.is_nan()));
    let dr = compute_dr_df_matrices_cov(
        &spec(),
        &IPREDS,
        &[1; 6],
        &times(),
        &[],
        &[],
        &[],
        &[],
        &SIGMA,
        c,
        None,
    );
    assert!(dr.iter().all(|m| m.iter().all(|v| v.is_nan())));
    // No observations: nothing to be unassigned.
    let empty = compute_r_matrix_with_correlations_cov(
        &spec(),
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &SIGMA,
        c,
    );
    assert_eq!(empty.len(), 0);
}

fn subject_with(arms: &[Option<f64>]) -> Subject {
    let obs_covariates = arms
        .iter()
        .map(|a| match a {
            Some(v) => HashMap::from([("ARM".to_string(), *v)]),
            None => HashMap::new(),
        })
        .collect();
    Subject {
        obs_times: (0..arms.len()).map(|j| j as f64).collect(),
        observations: vec![1.0; arms.len()],
        obs_cmts: vec![1; arms.len()],
        obs_covariates,
        ..Default::default()
    }
}

fn serial_model(by: &[&str]) -> crate::types::CompiledModel {
    let mut model =
        crate::types::test_helpers::analytical_model(crate::types::GradientMethod::Auto);
    model.residual_serial = Some(crate::types::ResidualSerial {
        kind: ResidualSerialKind::Cs,
        rho_theta: 0,
        by: by.iter().map(|s| s.to_string()).collect(),
    });
    model
}

/// Blocks are numbered from 1 in first-row order; a row with a missing `by` value
/// is in none.
#[test]
fn residual_groups_reads_the_by_column_per_row() {
    let model = serial_model(&["ARM"]);
    let s = subject_with(&[
        Some(7.0),
        Some(3.0),
        None,
        Some(7.0),
        Some(f64::NAN),
        Some(3.0),
    ]);
    assert_eq!(residual_groups(&model, &s), vec![1, 2, 0, 1, 0, 2]);
    // No `correlation` line: no blocks at all.
    let mut plain = model;
    plain.residual_serial = None;
    assert!(residual_groups(&plain, &s).is_empty());
}

/// Two `by` columns: a block is the combined value, not either column alone.
#[test]
fn residual_groups_combines_several_by_columns() {
    let model = serial_model(&["ARM", "PER"]);
    let mut s = subject_with(&[Some(1.0), Some(1.0), Some(2.0), Some(2.0)]);
    for (j, per) in [1.0, 2.0, 1.0, 1.0].iter().enumerate() {
        s.obs_covariates[j].insert("PER".to_string(), *per);
    }
    assert_eq!(residual_groups(&model, &s), vec![1, 2, 3, 3]);
}

/// The routing predicate straddles: a subject with a two-row block couples rows
/// (FD outer gradient, #1536), one whose blocks are all singletons does not.
/// Value-free: true at ρ = 0 too.
#[test]
fn has_cross_observation_residual_sees_a_serial_block() {
    let model = serial_model(&["ARM"]);
    let mut paired = subject_with(&[Some(1.0), Some(2.0), Some(1.0)]);
    paired.obs_rgroup = residual_groups(&model, &paired);
    let mut singletons = subject_with(&[Some(1.0), Some(2.0), Some(3.0)]);
    singletons.obs_rgroup = residual_groups(&model, &singletons);
    let es = ErrorSpec::Single(crate::types::ErrorModel::Additive);
    for rho in [0.0, 0.4] {
        let c = cov(ResidualSerialKind::Cs, rho);
        assert!(has_cross_observation_residual(&es, &paired, &[1.0], c));
        assert!(!has_cross_observation_residual(&es, &singletons, &[1.0], c));
    }
    // Without the serial source the paired subject is independent.
    assert!(!has_cross_observation_residual(
        &es,
        &paired,
        &[1.0],
        ResidualCov::block_sigma(&[])
    ));
    // Unassigned blocks route to FD too (their `R` is NaN).
    let unassigned = subject_with(&[Some(1.0), Some(2.0), Some(3.0)]);
    assert!(has_cross_observation_residual(
        &es,
        &unassigned,
        &[1.0],
        cov(ResidualSerialKind::Cs, 0.4)
    ));
}
