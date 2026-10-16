use super::*;
use crate::parser::model_parser::parse_model_string;
use nalgebra::DMatrix;

/// Two levels, `A` (by STUDY, in ID) and `B` (by ARM, in A), and three kappas declared
/// out of level order — `K0` at B, `K1` at A, `K2` at B — so a level's κ positions are
/// not its declaration positions.
fn model() -> CompiledModel {
    parse_model_string(
        "[parameters]\n  theta T(1.0)\n  omega E ~ 0.1\n  level A by STUDY in ID\n  \
         level B by ARM in A\n  kappa K0 ~ 0.2 at B\n  kappa K1 ~ 0.3 at A\n  \
         kappa K2 ~ 0.4 at B\n  sigma S ~ 0.1\n[individual_parameters]\n  \
         P = T * exp(E + K0 + K1 + K2)\n[structural_model]\n  y = P\n[error_model]\n  \
         DV ~ additive(S)\n",
    )
    .expect("parse")
}

fn snap(study: f64, arm: f64) -> HashMap<String, f64> {
    [("STUDY".to_string(), study), ("ARM".to_string(), arm)]
        .into_iter()
        .collect()
}

/// Six observations over two studies with arms numbered 1, 2 inside each study, plus a
/// pk-only row in study 2, arm 1, and one in an arm no observation reaches (study 1,
/// arm 3).
fn population() -> Population {
    let rows = [
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, 2.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (2.0, 2.0),
    ];
    let subject = Subject {
        id: "1".to_string(),
        obs_times: (0..rows.len()).map(|j| j as f64 + 1.0).collect(),
        observations: vec![1.0; rows.len()],
        obs_cmts: vec![1; rows.len()],
        cens: vec![0; rows.len()],
        covariates: snap(1.0, 1.0),
        obs_covariates: rows.iter().map(|&(s, a)| snap(s, a)).collect(),
        pk_only_times: vec![3.5, 0.5],
        pk_only_covariates: vec![snap(2.0, 1.0), snap(1.0, 3.0)],
        ..Default::default()
    };
    Population {
        subjects: vec![subject],
        covariate_names: vec!["STUDY".to_string(), "ARM".to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

fn prepared() -> (CompiledModel, Population) {
    let m = model();
    let mut p = population();
    assign_level_cells(&m, &mut p);
    (m, p)
}

#[test]
fn every_record_gets_the_id_of_its_full_label_path() {
    let (_, p) = prepared();
    let s = &p.subjects[0];
    // Paths in record order: (1,1) (1,1) (1,2) (2,1) (2,2) (2,2), then the pk-only rows
    // (2,1) and (1,3). ARM = 1 in study 2 is a new cell, not study 1's arm 1.
    assert_eq!(s.occasions, vec![1, 1, 2, 3, 4, 4]);
    assert_eq!(s.pk_only_occasions, vec![3, 5]);
    assert!(s.dose_occasions.is_empty() && s.reset_occasions.is_empty());
}

#[test]
fn a_record_with_a_missing_label_gets_cell_zero() {
    let m = model();
    let mut p = population();
    p.subjects[0].obs_covariates[2].insert("ARM".to_string(), f64::NAN);
    p.subjects[0].obs_covariates[4].remove("STUDY");
    assign_level_cells(&m, &mut p);
    assert_eq!(p.subjects[0].occasions, vec![1, 1, 0, 2, 0, 3]);
}

/// The compact index of every (cell, κ), written out literally from the layout rule:
/// `[η | A: study 1, study 2 | B: (1,1) (1,2) (2,1) (2,2) (1,3), two κ each]`. Level A
/// holds K1 alone, level B holds K0 then K2.
#[test]
fn compact_indices_follow_the_level_blocks_index_for_index() {
    let (m, p) = prepared();
    let n = NestedLayout::of(&m, &p.subjects[0]).expect("a nested model");
    // Cells in `iov_occasion_groups` order: the observation cells 1..4, then the
    // pk-only-only cell 5 (study 1, arm 3).
    assert_eq!(n.n_cells(), 5);
    assert_eq!(n.n_compact(), 1 + 2 + 5 * 2);
    // (cell, [K0, K1, K2])
    let want: [(usize, [usize; 3]); 5] = [
        (0, [3, 1, 4]),   // study 1, arm 1
        (1, [5, 1, 6]),   // study 1, arm 2
        (2, [7, 2, 8]),   // study 2, arm 1
        (3, [9, 2, 10]),  // study 2, arm 2
        (4, [11, 1, 12]), // study 1, arm 3
    ];
    for (g, ks) in want {
        for (k, &idx) in ks.iter().enumerate() {
            assert_eq!(n.compact_index(g, k), idx, "cell {g}, κ {k}");
        }
    }
}

#[test]
fn expand_and_compact_are_inverse() {
    let (m, p) = prepared();
    let n = NestedLayout::of(&m, &p.subjects[0]).unwrap();
    let b: Vec<f64> = (0..n.n_compact()).map(|i| 10.0 + i as f64).collect();
    let cells = n.expand(&b);
    // Study 1's κ (index 1) is shared by its three arms; study 2's by its two.
    assert_eq!(cells[0], vec![13.0, 11.0, 14.0]);
    assert_eq!(cells[1][1], 11.0);
    assert_eq!(cells[4][1], 11.0);
    assert_eq!(cells[2][1], 12.0);
    assert_eq!(n.compact(&b[..1], &cells).unwrap(), b);
    // A vector per cell is required.
    assert!(n.compact(&b[..1], &cells[..4]).is_none());
}

fn omega_iov() -> DMatrix<f64> {
    // K0 and K2 (both at B) covary; nothing crosses levels.
    DMatrix::from_row_slice(3, 3, &[0.2, 0.0, 0.05, 0.0, 0.3, 0.0, 0.05, 0.0, 0.4])
}

#[test]
fn the_compact_prior_is_one_block_per_group_of_each_level() {
    let (m, p) = prepared();
    let n = NestedLayout::of(&m, &p.subjects[0]).unwrap();
    let omega = DMatrix::from_element(1, 1, 0.1);
    let sigma = n.prior(&omega, &omega_iov());
    let mut want = DMatrix::zeros(13, 13);
    want[(0, 0)] = 0.1;
    want[(1, 1)] = 0.3;
    want[(2, 2)] = 0.3;
    for g in 0..5 {
        let o = 3 + 2 * g;
        want[(o, o)] = 0.2;
        want[(o + 1, o + 1)] = 0.4;
        want[(o, o + 1)] = 0.05;
        want[(o + 1, o)] = 0.05;
    }
    assert_eq!(sigma, want);
}

/// `Σ_L Σ_g κ_gᵀΩ_L⁻¹κ_g + G_L·log|Ω_L|`, computed here from each level's own block, and
/// what `kappa_prior` reads off the full `Ω_IOV`'s inverse and Cholesky factor. A prior
/// summed over the five cells instead would count study 1's κ three times.
#[test]
fn the_kappa_prior_counts_each_group_once_against_its_own_level() {
    let (m, p) = prepared();
    let n = NestedLayout::of(&m, &p.subjects[0]).unwrap();
    let full = omega_iov();
    let inv = full.clone().try_inverse().unwrap();
    let chol = full.clone().cholesky().unwrap().l();
    let b: Vec<f64> = (0..n.n_compact()).map(|i| 0.1 * i as f64 - 0.4).collect();
    let got = n.kappa_prior(&b, &inv, &chol);

    let omega_a = DMatrix::from_element(1, 1, 0.3);
    let omega_b = DMatrix::from_row_slice(2, 2, &[0.2, 0.05, 0.05, 0.4]);
    let term = |o: &DMatrix<f64>, v: &[f64]| {
        let v = nalgebra::DVector::from_column_slice(v);
        v.dot(&(o.clone().try_inverse().unwrap() * &v)) + o.determinant().ln()
    };
    let mut want = 0.0;
    for g in 0..2 {
        want += term(&omega_a, &b[1 + g..2 + g]);
    }
    for g in 0..5 {
        want += term(&omega_b, &b[3 + 2 * g..5 + 2 * g]);
    }
    assert!(got.is_finite());
    assert!(
        (got - want).abs() < 1e-12 * want.abs().max(1.0),
        "{got} vs {want}"
    );
}

#[test]
fn a_model_without_nested_levels_has_no_nested_layout() {
    let m = parse_model_string(
        "[parameters]\n  theta T(1.0)\n  omega E ~ 0.1\n  kappa K ~ 0.2\n  sigma S ~ 0.1\n\
         [individual_parameters]\n  P = T * exp(E + K)\n[structural_model]\n  y = P\n\
         [error_model]\n  DV ~ additive(S)\n",
    )
    .unwrap();
    let mut p = population();
    let before = p.subjects[0].occasions.clone();
    assign_level_cells(&m, &mut p);
    assert_eq!(
        p.subjects[0].occasions, before,
        "a no-op for a one-level model"
    );
    assert!(NestedLayout::of(&m, &p.subjects[0]).is_none());
}

#[test]
fn obs_groups_at_keys_a_group_by_its_whole_path() {
    let (m, p) = prepared();
    let levels = m.nested_levels().unwrap();
    let a = obs_groups_at(levels, &p.subjects[0], 0);
    assert_eq!(
        a,
        vec![("A 1".into(), vec![0, 1, 2]), ("A 2".into(), vec![3, 4, 5])]
    );
    let b = obs_groups_at(levels, &p.subjects[0], 1);
    // Four arms, though only two arm labels: study 2's arm 1 is its own group.
    assert_eq!(b.len(), 4);
    // ...and its label says which study: `B 1` alone would name an arm in both.
    assert_eq!(b[2], ("A 2 / B 1".into(), vec![3]));
}
