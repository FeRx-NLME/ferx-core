//! `correlation cs|ar1|car1(RHO) by COL` — the serial residual blocks of #1081.
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
        ResidualSerialKind::Car1 => rho.powf((ROWS[j].1 - ROWS[k].1).abs()),
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

/// T1: every entry of `R` equals `C_jk(ρ) · (f_j f_k σ_p² + w_j w_k σ_a²)`, for
/// every structure and both signs of ρ (`car1`: two positive ρ, its admissible
/// range). Dies on: an off-diagonal without the weight (unscaled loadings), a block
/// spanning the two arms, AR(1) lags taken in row order instead of time order, CS
/// read as AR(1) or the reverse, `car1` lags taken as visit positions instead of
/// time gaps (the visits are irregular: 0, 2, 12 and 0, 1, 5).
#[test]
fn serial_r_matches_the_closed_form() {
    for kind in [
        ResidualSerialKind::Cs,
        ResidualSerialKind::Ar1,
        ResidualSerialKind::Car1,
    ] {
        let rhos = if kind == ResidualSerialKind::Car1 {
            [0.6, 0.3]
        } else {
            [0.6, -0.3]
        };
        for rho in rhos {
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
    // And `car1` straddles `ar1` on that pair: one visit apart but 10 time units,
    // so ρ^10, not ρ^1 — by hand, not through `closed_form_c`.
    assert_eq!((ROWS[0].1 - ROWS[4].1).abs(), 10.0);
    let r = scaled_r(ResidualSerialKind::Car1, 0.6);
    assert_relative_eq!(r[(0, 4)], 0.6f64.powi(10) * base04, max_relative = 1e-13);
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
    for kind in [
        ResidualSerialKind::Cs,
        ResidualSerialKind::Ar1,
        ResidualSerialKind::Car1,
    ] {
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
/// bare and scaled, every structure. Combined error, so the proportional slot's
/// slope loading is live — an additive-only fixture has `∂R/∂f ≡ 0` off the
/// diagonal and could not see a missing serial term.
#[test]
fn serial_dr_and_d2r_match_finite_differences() {
    let n = 6;
    for kind in [
        ResidualSerialKind::Cs,
        ResidualSerialKind::Ar1,
        ResidualSerialKind::Car1,
    ] {
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

/// The decorrelated autocorrelation statistic (#1081 follow-up) is the plain one
/// taken on `L⁻¹ (y − f)`: build residuals `e = L z` from a chosen `z` and the
/// subject's own `R`, hand the function their IWRES, and it must return the
/// statistic of `z` — while the raw IWRES give a different one, so the test sees
/// the whitening and not a pass-through.
#[test]
fn whitened_autocorrelation_is_the_statistic_of_l_inverse_residuals() {
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVY(10.0, 1.0, 100.0)\n  theta RHO(0.7) FIX\n  sigma ADD_ERR ~ 1.5 (sd)\n\
         [individual_parameters]\n  Y0 = TVY\n[structural_model]\n  y = Y0\n[error_model]\n  \
         DV ~ additive(ADD_ERR)\n  correlation ar1(RHO) by ARM\n",
    )
    .expect("parses");
    let params = model.default_params.clone();
    let mut s = subject_with(&[Some(1.0), Some(1.0), Some(2.0), Some(1.0), Some(2.0)]);
    s.obs_rgroup = residual_groups(&model, &s);
    let ipred = vec![10.0; 5];
    let r = r_matrix_maybe_scaled(
        &model.error_spec,
        &ipred,
        &[1; 5],
        &s,
        &params.sigma.values,
        model.residual_cov_at(&params),
        None,
    );
    let l = r.clone().cholesky().expect("PD").l();
    let z = nalgebra::DVector::from_vec(vec![0.8, -0.3, 1.1, 0.4, -0.9]);
    let e = &l * &z;
    s.observations = (0..5).map(|j| ipred[j] + e[j]).collect();
    let mut res = crate::types::test_helpers::dummy_subject("1", 0, 5);
    res.ipred = ipred.clone();
    res.iwres = (0..5).map(|j| e[j] / r[(j, j)].sqrt()).collect();
    let pop = crate::types::Population {
        subjects: vec![s],
        covariate_names: vec!["ARM".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let got = whitened_residual_autocorrelation(&model, &pop, &params, std::slice::from_ref(&res));
    let want = residual_autocorrelation(std::iter::once(z.as_slice()));
    assert!(got.0.is_finite() && got.1.is_finite(), "{got:?}");
    assert_relative_eq!(got.0, want.0, max_relative = 1e-12);
    assert_relative_eq!(got.1, want.1, max_relative = 1e-12);
    let raw = iwres_autocorrelation(std::slice::from_ref(&res));
    assert!(
        (raw.1 - want.1).abs() > 0.1,
        "raw {raw:?} vs whitened {want:?}"
    );
}

/// The two dense derivative families at this file's fixture, for one structure.
fn dense_families(
    kind: ResidualSerialKind,
    rho: f64,
) -> (Vec<DMatrix<f64>>, Vec<Vec<DMatrix<f64>>>) {
    let m = mult();
    let dr = compute_dr_df_matrices_cov(
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
        Some(&m),
    );
    let d2 = compute_d2r_df2_matrices_cov(
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
        Some(&m),
    );
    (dr, d2)
}

fn dc_at(kind: ResidualSerialKind, rho: f64) -> DMatrix<f64> {
    compute_dr_df_columns_cov(
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
        Some(&mult()),
    )
}

/// #1960: the column / compact forms the dense outer gradient reads hold **every**
/// nonzero of the dense `∂R/∂f` and `∂²R/∂f²` families — bit for bit, since both are
/// filled by one visitor — and the dense families have exactly the row-and-column
/// pattern those forms assume (anything outside it would be silently dropped).
#[test]
fn compact_derivative_forms_equal_the_dense_builders() {
    for kind in [
        ResidualSerialKind::Cs,
        ResidualSerialKind::Ar1,
        ResidualSerialKind::Car1,
    ] {
        let (dr, d2) = dense_families(kind, 0.4);
        let dc = dc_at(kind, 0.4);
        let m = mult();
        let (x, y) = compute_d2r_df2_compact_cov(
            &spec(),
            &IPREDS,
            &[1; 6],
            &times(),
            &[],
            &[],
            &[],
            &rgroup(),
            &SIGMA,
            cov(kind, 0.4),
            Some(&m),
        );
        let n = IPREDS.len();
        let mut off_diagonal = 0usize;
        for a in 0..n {
            for i in 0..n {
                for j in 0..n {
                    // ∂R/∂f_a from its column.
                    let want = if i == a {
                        dc[(j, a)]
                    } else if j == a {
                        dc[(i, a)]
                    } else {
                        0.0
                    };
                    assert_eq!(dr[a][(i, j)], want, "{kind:?}: dR/df_{a} at ({i},{j})");
                    if i != j && want != 0.0 {
                        off_diagonal += 1;
                    }
                    for b in 0..n {
                        let want = if a != b {
                            if (i, j) == (a, b) || (i, j) == (b, a) {
                                x[(a, b)]
                            } else {
                                0.0
                            }
                        } else if i == a {
                            y[(j, a)]
                        } else if j == a {
                            y[(i, a)]
                        } else {
                            0.0
                        };
                        assert_eq!(
                            d2[a][b][(i, j)],
                            want,
                            "{kind:?}: d2R/df_{a}df_{b} ({i},{j})"
                        );
                    }
                }
            }
        }
        assert!(off_diagonal > 0, "{kind:?}: fixture must couple rows");
    }
}

/// #1960: `serial_rho_derivatives` is the ρ-derivative of the builders — of `R` and of
/// the `∂R/∂f` column form — against a Richardson central difference in ρ, for each
/// structure (`ar1` and `car1` are the non-linear `C_jk(ρ)`), and is zero on the
/// diagonal, which ρ never reaches.
#[test]
fn serial_rho_derivatives_match_fd_of_the_builders() {
    for kind in [
        ResidualSerialKind::Cs,
        ResidualSerialKind::Ar1,
        ResidualSerialKind::Car1,
    ] {
        let rho = 0.4;
        let (dr, ddc) = serial_rho_derivatives(
            &spec(),
            &IPREDS,
            &[1; 6],
            &times(),
            &[],
            &rgroup(),
            &SIGMA,
            cov(kind, rho),
            Some(&mult()),
        )
        .expect("finite");
        let fd = |g: &dyn Fn(f64) -> DMatrix<f64>| {
            let h = 1e-4;
            let d1 = (g(rho + h) - g(rho - h)) / (2.0 * h);
            let d2 = (g(rho + h / 2.0) - g(rho - h / 2.0)) / h;
            (4.0 * d2 - d1) / 3.0
        };
        let want_r = fd(&|r| scaled_r(kind, r));
        let want_dc = fd(&|r| dc_at(kind, r));
        let n = IPREDS.len();
        for i in 0..n {
            assert_eq!(dr[(i, i)], 0.0, "{kind:?}: ρ never reaches the diagonal");
            for j in 0..n {
                assert_relative_eq!(
                    dr[(i, j)],
                    want_r[(i, j)],
                    max_relative = 1e-8,
                    epsilon = 1e-9
                );
                assert_relative_eq!(
                    ddc[(i, j)],
                    want_dc[(i, j)],
                    max_relative = 1e-8,
                    epsilon = 1e-9
                );
            }
        }
        assert!(dr.iter().any(|v| v.abs() > 1e-3), "{kind:?}: live");
        assert!(ddc.iter().any(|v| v.abs() > 1e-3), "{kind:?}: live");
    }
}

/// #1995 review: a row whose diagonal variance sits on the `MIN_VARIANCE` floor is
/// locally constant in `f`, so the dense `∂R/∂f` and `∂²R/∂f²` visitors drop its
/// diagonal self-terms (the #958 gate `ErrorSpec::dvar_df` applies on the diagonal
/// path), while a live row keeps them. Each column is checked against a central
/// difference in `f_m` of the builder it differentiates (`R` for the first family, the
/// `∂R/∂f` column form for the second), so a visitor that ignores the floor dies on
/// the floored row and one that floors every row dies on the live one. The raw slope
/// the gate removes is asserted nonzero, so the floored cell cannot pass by accident.
#[test]
fn dense_derivatives_drop_a_floored_diagonal() {
    let es = ErrorSpec::Single(crate::types::ErrorModel::Proportional);
    let sigma = [0.2];
    let c = cov(ResidualSerialKind::Ar1, 0.4);
    // Row 1 at f = 1e-7: (f·σ)² = 4e-16, under the floor; the others are live.
    let mut f = IPREDS;
    f[1] = 1e-7;
    let cmts = [1usize; 6];
    let r_at = |fv: &[f64]| {
        compute_r_matrix_with_correlations_cov(
            &es,
            fv,
            &cmts,
            &times(),
            &[],
            &[],
            &[],
            &rgroup(),
            &sigma,
            c,
        )
    };
    let dc_at = |fv: &[f64]| {
        compute_dr_df_columns_cov(
            &es,
            fv,
            &cmts,
            &times(),
            &[],
            &[],
            &[],
            &rgroup(),
            &sigma,
            c,
            None,
        )
    };
    assert_eq!(
        r_at(&f)[(1, 1)],
        MIN_VARIANCE,
        "fixture: row 1 must be floored"
    );
    assert!(
        r_at(&f)[(0, 0)] > MIN_VARIANCE,
        "fixture: row 0 must be live"
    );
    let dc = dc_at(&f);
    let (_, y) = compute_d2r_df2_compact_cov(
        &es,
        &f,
        &cmts,
        &times(),
        &[],
        &[],
        &[],
        &rgroup(),
        &sigma,
        c,
        None,
    );
    // What an ungated diagonal would hold on the floored row: 2fσ² (and 2σ²).
    assert!(es.dvar_df_raw(1, f[1], &sigma) > 0.0);
    assert_eq!(dc[(1, 1)], 0.0, "floored row keeps a ∂R_11/∂f_1");
    assert_eq!(y[(1, 1)], 0.0, "floored row keeps a ∂²R_11/∂f_1²");
    for m in [0usize, 1] {
        let h = 1e-3 * f[m];
        let (mut fp, mut fm) = (f, f);
        fp[m] += h;
        fm[m] -= h;
        let fd_r = (r_at(&fp) - r_at(&fm)) / (2.0 * h);
        let fd_dc = (dc_at(&fp) - dc_at(&fm)) / (2.0 * h);
        for k in 0..f.len() {
            let tol = |w: f64| 1e-7 * w.abs().max(1e-12);
            assert!(
                (dc[(k, m)] - fd_r[(k, m)]).abs() <= tol(fd_r[(k, m)]),
                "∂R_{k}{m}/∂f_{m}: {} vs FD {}",
                dc[(k, m)],
                fd_r[(k, m)]
            );
            assert!(
                (y[(k, m)] - fd_dc[(k, m)]).abs() <= tol(fd_dc[(k, m)]),
                "∂²R_{k}{m}/∂f_{m}²: {} vs FD {}",
                y[(k, m)],
                fd_dc[(k, m)]
            );
        }
    }
    assert!(
        dc[(0, 0)] > 0.0 && y[(0, 0)] > 0.0,
        "live row lost its diagonal terms"
    );
}
