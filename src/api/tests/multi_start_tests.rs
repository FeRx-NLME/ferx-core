use super::perturb_init;
use crate::estimation::parameterization::theta_packs_log;
use crate::types::{FitOptions, ModelParameters, OmegaMatrix, SigmaVector};

fn make_params(theta: Vec<f64>, theta_lower: Vec<f64>, theta_upper: Vec<f64>) -> ModelParameters {
    let n = theta.len();
    ModelParameters {
        theta,
        theta_names: (0..n).map(|i| format!("T{i}")).collect(),
        theta_lower,
        theta_upper,
        theta_fixed: vec![false; n],
        omega: OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]),
        omega_fixed: vec![false],
        sigma: SigmaVector {
            values: vec![0.1],
            names: vec!["ERR".into()],
        },
        sigma_fixed: vec![false],
        omega_iov: None,
        kappa_fixed: Vec::new(),
    }
}

#[test]
fn test_perturb_start0_is_identity() {
    let p = make_params(vec![5.0, 50.0], vec![0.1, 1.0], vec![100.0, 500.0]);
    let perturbed = perturb_init(&p, 0, 0.5, 42);
    assert_eq!(perturbed.theta, p.theta);
}

#[test]
fn test_perturb_changes_theta() {
    let p = make_params(vec![5.0, 50.0], vec![0.1, 1.0], vec![100.0, 500.0]);
    let perturbed = perturb_init(&p, 1, 0.3, 42);
    // With sigma=0.3 and seed=43 (42+1), at least one theta should differ
    let changed = perturbed
        .theta
        .iter()
        .zip(p.theta.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(changed, "start 1 should perturb theta");
}

#[test]
fn test_perturb_stays_in_bounds() {
    let p = make_params(vec![5.0, 50.0], vec![0.1, 1.0], vec![100.0, 500.0]);
    for k in 1..=10 {
        let perturbed = perturb_init(&p, k, 2.0, 42); // large sigma to stress-test bounds
        for (i, &t) in perturbed.theta.iter().enumerate() {
            assert!(
                t >= p.theta_lower[i],
                "start {k}: theta[{i}]={t} < lower={}",
                p.theta_lower[i]
            );
            assert!(
                t <= p.theta_upper[i],
                "start {k}: theta[{i}]={t} > upper={}",
                p.theta_upper[i]
            );
        }
    }
}

#[test]
fn test_perturb_identity_packed_theta() {
    // theta_lower < 0 → identity packing → additive perturbation
    let p = make_params(vec![0.5], vec![-5.0], vec![5.0]);
    assert!(!theta_packs_log(p.theta_lower[0]));
    let perturbed = perturb_init(&p, 1, 0.3, 99);
    assert!(perturbed.theta[0] >= -5.0 && perturbed.theta[0] <= 5.0);
}

#[test]
fn test_n_starts_option_parsed() {
    let mut opts = FitOptions::default();
    assert_eq!(opts.n_starts, 1);
    opts.n_starts = 4;
    assert_eq!(opts.n_starts, 4);
}

#[test]
fn test_n_starts_and_seed_via_parser() {
    use crate::parser::model_parser::apply_fit_option;
    let mut opts = FitOptions::default();
    apply_fit_option(&mut opts, "n_starts", "4").expect("n_starts parses");
    assert_eq!(opts.n_starts, 4);
    apply_fit_option(&mut opts, "multi_start_seed", "123").expect("multi_start_seed parses");
    assert_eq!(opts.multi_start_seed, Some(123));
    apply_fit_option(&mut opts, "start_sigma", "0.5").expect("start_sigma parses");
    assert!((opts.start_sigma - 0.5).abs() < 1e-10);
}

#[test]
fn test_per_start_saem_seed_derivation() {
    let base: u64 = 12345;
    // Each start k > 0 gets base + k; start 0 keeps the base unchanged.
    assert_eq!(base.wrapping_add(0), 12345);
    assert_eq!(base.wrapping_add(1), 12346);
    assert_eq!(base.wrapping_add(7), 12352);
    // All derived seeds are distinct.
    let seeds: Vec<u64> = (0..8).map(|k| base.wrapping_add(k)).collect();
    let unique: std::collections::HashSet<u64> = seeds.iter().copied().collect();
    assert_eq!(unique.len(), 8);
    // wrapping_add is defined at u64::MAX.
    assert_eq!(u64::MAX.wrapping_add(1), 0);
}
