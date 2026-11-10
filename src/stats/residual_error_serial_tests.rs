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

/// #2656: with `iiv_on_ruv` the whitening is still `L⁻¹` of the subject's own `R`. IWRES
/// already divides by `s = exp(η̂_ruv)` and the function builds `R` at η = 0, and the two
/// cancel: from residuals `e = s·L₀ z` and those IWRES it must return the statistic of `z`.
#[test]
fn whitened_autocorrelation_reads_the_subjects_own_residual_scale() {
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVY(10.0, 1.0, 100.0)\n  theta RHO(0.7) FIX\n  omega ETA_RUV ~ 0.2\n  \
         sigma ADD_ERR ~ 1.5 (sd)\n[individual_parameters]\n  Y0 = TVY\n[structural_model]\n  \
         y = Y0\n[error_model]\n  DV ~ additive(ADD_ERR)\n  correlation ar1(RHO) by ARM\n  \
         iiv_on_ruv = ETA_RUV\n",
    )
    .expect("parses");
    let params = model.default_params.clone();
    let mut s = subject_with(&[Some(1.0), Some(1.0), Some(2.0), Some(1.0), Some(2.0)]);
    s.obs_rgroup = residual_groups(&model, &s);
    let ipred = vec![10.0; 5];
    let r0 = r_matrix_maybe_scaled(
        &model.error_spec,
        &ipred,
        &[1; 5],
        &s,
        &params.sigma.values,
        model.residual_cov_at(&params),
        None,
    );
    let sd = 1.8_f64; // exp(η̂_ruv)
    let z = nalgebra::DVector::from_vec(vec![0.8, -0.3, 1.1, 0.4, -0.9]);
    let e = r0.clone().cholesky().expect("PD").l() * &z * sd;
    s.observations = (0..5).map(|j| ipred[j] + e[j]).collect();
    let mut res = crate::types::test_helpers::dummy_subject("1", 0, 5);
    res.ipred = ipred.clone();
    res.iwres = (0..5).map(|j| e[j] / (r0[(j, j)].sqrt() * sd)).collect();
    let pop = crate::types::Population {
        subjects: vec![s],
        covariate_names: vec!["ARM".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let got = whitened_residual_autocorrelation(&model, &pop, &params, std::slice::from_ref(&res));
    // Split by block (#2798): ARM [1, 1, 2, 1, 2] is rows {0, 1, 3} and {2, 4}.
    let blocks = [vec![z[0], z[1], z[3]], vec![z[2], z[4]]];
    let want = residual_autocorrelation(blocks.iter().map(Vec::as_slice));
    assert!(got.0.is_finite() && got.1.is_finite(), "{got:?}");
    assert_relative_eq!(got.0, want.0, max_relative = 1e-12);
    assert_relative_eq!(got.1, want.1, max_relative = 1e-12);
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
    // Split by block (#2798): ARM [1, 1, 2, 1, 2] is rows {0, 1, 3} and {2, 4}.
    let blocks = [vec![z[0], z[1], z[3]], vec![z[2], z[4]]];
    let want = residual_autocorrelation(blocks.iter().map(Vec::as_slice));
    assert!(got.0.is_finite() && got.1.is_finite(), "{got:?}");
    assert_relative_eq!(got.0, want.0, max_relative = 1e-12);
    assert_relative_eq!(got.1, want.1, max_relative = 1e-12);
    let raw = iwres_autocorrelation(std::slice::from_ref(&res));
    assert!(
        (raw.1 - want.1).abs() > 0.1,
        "raw {raw:?} vs whitened {want:?}"
    );
    // B4 (#2798), by hand: the blocks [0.8, −0.3, 0.4] and [1.1, −0.9] have squared
    // steps 1.21 + 0.49 + 4.00 = 5.70 over pair means 0.365 + 0.125 + 1.01 = 1.5, so
    // DW = 3.8. The whole series would pair across the blocks: 5.35 / 2.185 = 2.4485.
    // This fixture has no `occasions`, so the split here is `obs_rgroup` alone.
    assert!(pop.subjects[0].occasions.is_empty());
    assert!(
        (got.1 - 3.8).abs() < 1e-12,
        "whitened DW split by residual block: 3.8, got {}",
        got.1
    );
}

/// B5 (#2798): rows are grouped by key, not by contiguous run. Two occasions written
/// interleaved, `[1, 2, 1, 2]`, give the same series — and the same statistic — as the
/// same rows written contiguously, `[1, 1, 2, 2]`, with the values reordered to match.
/// Each occasion's values are `[1, 2]` and `[−1, 3]`: steps 1 and 16 over pair means
/// 2.5 and 5, DW = 17 / 7.5.
///
///
/// Both layouts are parallel (#2829): the interleaved one because its labels come
/// back, the contiguous one because its clock restarts at the second occasion.
///
/// Mutation that must redden this: grouping by contiguous run of the key (the
/// interleaved layout would then be four one-row series and no pair at all).
#[test]
fn grouped_dw_groups_by_key_not_by_contiguous_run() {
    let pop_of = |occasions: Vec<u32>, obs_times: Vec<f64>| crate::types::Population {
        subjects: vec![Subject {
            obs_times,
            observations: vec![0.0; 4],
            obs_cmts: vec![1; 4],
            occasions,
            ..Default::default()
        }],
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let res_of = |iwres: Vec<f64>| {
        let mut r = crate::types::test_helpers::dummy_subject("1", 0, 4);
        r.iwres = iwres;
        r
    };
    let interleaved = grouped_iwres_autocorrelation(
        &pop_of(vec![1, 2, 1, 2], vec![0.0, 0.0, 1.0, 1.0]),
        &[res_of(vec![1.0, -1.0, 2.0, 3.0])],
    );
    let contiguous = grouped_iwres_autocorrelation(
        &pop_of(vec![1, 1, 2, 2], vec![0.0, 1.0, 0.0, 1.0]),
        &[res_of(vec![1.0, 2.0, -1.0, 3.0])],
    );
    assert!(interleaved.1.is_finite() && contiguous.1.is_finite());
    assert!(
        (contiguous.1 - 17.0 / 7.5).abs() < 1e-12,
        "contiguous: {}",
        contiguous.1
    );
    assert_eq!(interleaved.1.to_bits(), contiguous.1.to_bits());
    assert_eq!(interleaved.0.to_bits(), contiguous.0.to_bits());
}

/// #2798 review row 2: the key is the **pair** `(occasion, residual block)`, not
/// either part alone, nor occasions-if-present-else-block. Two occasions crossed with
/// two blocks give four cells of two rows each; every other key joins rows across a
/// cell boundary and lands on a different value, which the test computes and asserts
/// apart (so the straddle cannot quietly become a tautology).
///
/// Rows `e = [1, 2 | 3, 1 | −1, 1 | 2, 2]`, cells `(1,0) (1,1) (2,0) (2,1)`:
/// pairs (1,2) (3,1) (−1,1) (2,2), Σ diff² = 1 + 4 + 4 + 0 = 9,
/// Σ (eₜ²+eₜ₋₁²)/2 = 2.5 + 5 + 1 + 4 = 12.5, DW = 0.72.
/// Occasion alone: 11 / 21.5 = 0.5116. Block alone: 19 / 17.5 = 1.0857.
#[test]
fn grouped_dw_keys_on_occasion_and_block_together() {
    // The two occasions are parallel: each restarts the clock (#2829).
    let pop_of = |occasions: Vec<u32>, obs_rgroup: Vec<u32>| crate::types::Population {
        subjects: vec![Subject {
            obs_times: (0..8).map(|j| f64::from(j % 4)).collect(),
            observations: vec![0.0; 8],
            obs_cmts: vec![1; 8],
            occasions,
            obs_rgroup,
            ..Default::default()
        }],
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let mut res = crate::types::test_helpers::dummy_subject("1", 0, 8);
    res.iwres = vec![1.0, 2.0, 3.0, 1.0, -1.0, 1.0, 2.0, 2.0];
    let res = [res];
    let occ = vec![1, 1, 1, 1, 2, 2, 2, 2];
    let grp = vec![0, 0, 1, 1, 0, 0, 1, 1];
    assert!(!occasions_run_in_time(
        &pop_of(occ.clone(), grp.clone()).subjects[0]
    ));
    let both = grouped_iwres_autocorrelation(&pop_of(occ.clone(), grp.clone()), &res).1;
    let occ_only = grouped_iwres_autocorrelation(&pop_of(occ, vec![]), &res).1;
    let grp_only = grouped_iwres_autocorrelation(&pop_of(vec![], grp), &res).1;
    assert!(both.is_finite() && occ_only.is_finite() && grp_only.is_finite());
    assert!(
        (occ_only - 11.0 / 21.5).abs() < 1e-12,
        "occasion only: {occ_only}"
    );
    assert!(
        (grp_only - 19.0 / 17.5).abs() < 1e-12,
        "block only: {grp_only}"
    );
    assert!(
        (both - 9.0 / 12.5).abs() < 1e-12,
        "(occasion, block): {both}, occasion alone {occ_only}, block alone {grp_only}"
    );
}

/// T7 (#2829): only the **occasion** part of the key is gated on the clock. The
/// fixture of the test above with the occasions in time (`obs_times = 0..8`): the
/// occasions are joined, the two residual blocks still split, so the statistic is
/// the block-alone one, `19 / 17.5`, bit for bit what the same rows give with no
/// occasions at all. The parallel twin (clock restarting) is `9 / 12.5`.
///
/// Mutation that must redden this: the residual-block part gated by the predicate
/// too (in time, nothing splits: the whole series, `15 / 22.5`).
#[test]
fn occasions_in_time_still_split_at_residual_blocks() {
    let pop_of = |occasions: Vec<u32>, obs_times: Vec<f64>| crate::types::Population {
        subjects: vec![Subject {
            obs_times,
            observations: vec![0.0; 8],
            obs_cmts: vec![1; 8],
            occasions,
            obs_rgroup: vec![0, 0, 1, 1, 0, 0, 1, 1],
            ..Default::default()
        }],
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let mut res = crate::types::test_helpers::dummy_subject("1", 0, 8);
    res.iwres = vec![1.0, 2.0, 3.0, 1.0, -1.0, 1.0, 2.0, 2.0];
    let res = [res];
    let occ = vec![1, 1, 1, 1, 2, 2, 2, 2];
    let running: Vec<f64> = (0..8).map(f64::from).collect();
    let restarting: Vec<f64> = (0..8).map(|j| f64::from(j % 4)).collect();
    let in_time = grouped_iwres_autocorrelation(&pop_of(occ.clone(), running.clone()), &res);
    let no_occ = grouped_iwres_autocorrelation(&pop_of(vec![], running), &res);
    let parallel = grouped_iwres_autocorrelation(&pop_of(occ, restarting), &res);
    assert!(in_time.1.is_finite() && no_occ.1.is_finite() && parallel.1.is_finite());
    assert!(
        (in_time.1 - 19.0 / 17.5).abs() < 1e-12,
        "block only: {}",
        in_time.1
    );
    assert_eq!(in_time.1.to_bits(), no_occ.1.to_bits());
    assert_eq!(in_time.0.to_bits(), no_occ.0.to_bits());
    assert!(
        (parallel.1 - 9.0 / 12.5).abs() < 1e-12,
        "(occasion, block): {}",
        parallel.1
    );
}

/// T7b (#2829, plan amendment 1): the **whitened** statistic inherits the gate,
/// because it lives in `split_by_residual_group` and not in the raw entry. One
/// residual block (`ARM` constant) holding occasions `[1, 1, 2, 2]`: in time the
/// result is bit for bit the one with no occasions; with the clock restarted at the
/// boundary it differs.
///
/// The model is `cs`, whose `R` does not read the times, so the three fixtures
/// whiten to the same `w` and only the split can tell them apart.
///
/// Mutation that must redden this: the gate moved from `split_by_residual_group` to
/// `grouped_iwres_autocorrelation` (the whitened path then splits the in-time arm).
#[test]
fn whitened_dw_joins_occasions_that_run_in_time() {
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVY(10.0, 1.0, 100.0)\n  theta RHO(0.4) FIX\n  sigma ADD_ERR ~ 1.5 (sd)\n\
         [individual_parameters]\n  Y0 = TVY\n[structural_model]\n  y = Y0\n[error_model]\n  \
         DV ~ additive(ADD_ERR)\n  correlation cs(RHO) by ARM\n",
    )
    .expect("parses");
    let params = model.default_params.clone();
    let z = [0.8, -0.3, 1.1, 0.4];
    let run = |occasions: Vec<u32>, obs_times: Vec<f64>| {
        let mut s = subject_with(&[Some(1.0); 4]);
        s.obs_times = obs_times;
        s.obs_rgroup = residual_groups(&model, &s);
        assert_eq!(s.obs_rgroup, vec![1; 4], "one residual block");
        s.occasions = occasions;
        let ipred = vec![10.0; 4];
        let r = r_matrix_maybe_scaled(
            &model.error_spec,
            &ipred,
            &[1; 4],
            &s,
            &params.sigma.values,
            model.residual_cov_at(&params),
            None,
        );
        assert!(r[(0, 1)].abs() > 0.1, "the block must be correlated: {r}");
        let e = r.clone().cholesky().expect("PD").l() * nalgebra::DVector::from_row_slice(&z);
        s.observations = (0..4).map(|j| ipred[j] + e[j]).collect();
        let mut res = crate::types::test_helpers::dummy_subject("1", 0, 4);
        res.ipred = ipred;
        res.iwres = (0..4).map(|j| e[j] / r[(j, j)].sqrt()).collect();
        let pop = crate::types::Population {
            subjects: vec![s],
            covariate_names: vec!["ARM".into()],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        whitened_residual_autocorrelation(&model, &pop, &params, std::slice::from_ref(&res))
    };
    let running = vec![0.0, 1.0, 2.0, 3.0];
    let in_time = run(vec![1, 1, 2, 2], running.clone());
    let no_occ = run(vec![], running);
    let parallel = run(vec![1, 1, 2, 2], vec![0.0, 1.0, 0.0, 1.0]);
    assert!(in_time.1.is_finite() && no_occ.1.is_finite() && parallel.1.is_finite());
    assert_eq!(in_time.1.to_bits(), no_occ.1.to_bits(), "DW");
    assert_eq!(in_time.0.to_bits(), no_occ.0.to_bits(), "lag-1 r");
    // By hand on `z`: whole, steps 1.21 + 1.96 + 0.49 over 0.365 + 0.65 + 0.685;
    // split at the occasion, steps 1.21 + 0.49 over 0.365 + 0.685.
    assert_relative_eq!(in_time.1, 3.66 / 1.7, max_relative = 1e-12);
    assert_relative_eq!(parallel.1, 1.70 / 1.05, max_relative = 1e-12);
}
