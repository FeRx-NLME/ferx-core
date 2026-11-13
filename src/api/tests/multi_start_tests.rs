use super::perturb_init;
use crate::estimation::parameterization::theta_packs_log;
use crate::types::{FitOptions, ModelParameters, OmegaMatrix, SigmaVector};

fn make_params(theta: Vec<f64>, theta_lower: Vec<f64>, theta_upper: Vec<f64>) -> ModelParameters {
    let n = theta.len();
    ModelParameters {
        residual_correlations: Vec::new(),
        residual_correlation_fixed: Vec::new(),
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
        mixture: None,
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

// ── #2867: the per-start records ───────────────────────────────────────────────────────

use super::{multistart_prefers, select_start};
use crate::diagnostics::{Diagnostic, EngineError};
use crate::types::test_helpers::empty_fit_result;
use crate::types::{FitResult, FitStatus, StartRecord};

fn start_fit(ofv: f64, converged: bool, theta: f64) -> FitResult {
    FitResult {
        ofv,
        converged,
        theta: vec![theta],
        n_iterations: 7,
        wall_time_secs: 0.25,
        ..empty_fit_result()
    }
}

/// The selection as `fit_unstamped` ran it before #2867, kept here as the oracle: the
/// winner's index under `multistart_prefers`, first start winning a tie.
fn old_selection(results: &[(usize, Result<FitResult, EngineError>)]) -> Option<usize> {
    let mut best: Option<(usize, f64, bool)> = None;
    for (k, res) in results {
        if let Ok(r) = res {
            let better = match best {
                None => true,
                Some((_, o, c)) => multistart_prefers(o, c, r.ofv, r.converged),
            };
            if better {
                best = Some((*k, r.ofv, r.converged));
            }
        }
    }
    best.map(|b| b.0)
}

fn mixed_results() -> Vec<(usize, Result<FitResult, EngineError>)> {
    vec![
        (0, Ok(start_fit(12.0, true, 1.0))),
        (1, Err(EngineError::uncoded("start refused"))),
        (2, Ok(start_fit(10.0, false, 2.0))),
        (3, Ok(start_fit(11.0, true, 3.0))),
        (4, Ok(start_fit(f64::NAN, true, 4.0))),
        (5, Ok(start_fit(11.0, true, 5.0))),
    ]
}

/// One row per start in start order, `selected` on exactly the winner, and the winner is
/// the start the pre-#2867 loop picked: converged start 3 at 11.0, ahead of the lower but
/// not-converged start 2 and of start 5, which ties it and comes later.
#[test]
fn select_start_records_every_start_and_picks_what_the_ranking_picks() {
    let expected = old_selection(&mixed_results());
    assert_eq!(expected, Some(3), "premise: the ranking picks start 3");
    let (best, failed, starts) = select_start(mixed_results(), |r| r.ofv, false);
    let (k, winner) = best.expect("a start ran");
    assert_eq!(Some(k), expected);
    assert_eq!(winner.theta, vec![3.0]);
    assert_eq!(failed, vec!["start 1: start refused".to_string()]);

    assert_eq!(
        starts.iter().map(|s| s.start).collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5]
    );
    let selected: Vec<usize> = starts
        .iter()
        .filter(|s| s.selected)
        .map(|s| s.start)
        .collect();
    assert_eq!(selected, vec![3]);
    assert_eq!(starts[3].ofv, Some(winner.ofv));
    assert_eq!(starts[3].theta, winner.theta);
    assert_eq!(starts[3].status, Some(FitStatus::Converged));
    assert_eq!(
        (starts[3].n_iterations, starts[3].wall_time_secs),
        (7, 0.25)
    );
    assert!(matches!(
        starts[2].status,
        Some(FitStatus::NotConverged { .. })
    ));
    assert!(!starts[2].converged);
    // No penalty: no second column.
    assert!(starts.iter().all(|s| s.objective.is_none()));
    // A NaN OFV is recorded as absent, on a start that still ran.
    assert_eq!((starts[4].ofv, starts[4].converged), (None, true));
    assert!(starts[4].error.is_none());
}

/// A failed start keeps its message and its diagnostic code, and nothing a result would
/// carry. The code column is what #2073 reads.
#[test]
fn a_failed_start_keeps_its_message_and_code() {
    let coded = EngineError::from_diagnostic(Diagnostic::error("E_AGQ_NODES_ZERO", "no nodes"));
    assert_eq!(coded.code(), Some("E_AGQ_NODES_ZERO"), "premise");
    let text = coded.to_string();
    let (best, _, starts) = select_start(
        vec![(0, Err(coded)), (1, Ok(start_fit(5.0, true, 1.0)))],
        |r| r.ofv,
        false,
    );
    assert_eq!(best.map(|b| b.0), Some(1));
    assert_eq!(
        starts[0],
        StartRecord {
            start: 0,
            ofv: None,
            objective: None,
            converged: false,
            status: None,
            n_iterations: 0,
            wall_time_secs: 0.0,
            theta: Vec::new(),
            selected: false,
            error: Some(text),
            error_code: Some("E_AGQ_NODES_ZERO".into()),
        }
    );
    assert!(starts[1].selected);
    // An uncoded failure has the message and no code; with no start that ran, none is
    // selected.
    let (best, _, starts) = select_start(
        vec![(0, Err(EngineError::uncoded("boom")))],
        |r| r.ofv,
        false,
    );
    assert!(best.is_none());
    assert_eq!(starts[0].error.as_deref(), Some("boom"));
    assert_eq!(starts[0].error_code, None);
    assert!(!starts[0].selected);
}

/// Under a penalty the ranked value is recorded beside the OFV, and the winner is the
/// lowest *ranked* start: start 0 has the higher OFV and the lower penalized objective.
#[test]
fn select_start_records_the_ranked_objective_under_a_penalty() {
    let results = || {
        vec![
            (0, Ok(start_fit(-100.0, true, 0.5))),
            (1, Ok(start_fit(-105.0, true, 4.0))),
        ]
    };
    // penalty = theta², so start 1 pays 16 for its 5 OFV.
    let rank = |r: &FitResult| r.ofv + r.theta[0] * r.theta[0];
    let (best, _, starts) = select_start(results(), rank, true);
    assert_eq!(best.map(|b| b.0), Some(0));
    assert_eq!(starts[0].objective, Some(-99.75));
    assert_eq!(starts[1].objective, Some(-89.0));
    assert_eq!((starts[0].ofv, starts[1].ofv), (Some(-100.0), Some(-105.0)));
    // The same ranking without the flag picks the same start and writes no column.
    let (best, _, starts) = select_start(results(), rank, false);
    assert_eq!(best.map(|b| b.0), Some(0));
    assert!(starts.iter().all(|s| s.objective.is_none()));
}

/// The fit YAML's `multi_start:` block: the summary numbers and one row per start. Absent
/// for a fit with no rows, so a single-start YAML is unchanged.
#[test]
fn multi_start_yaml_block_lists_the_summary_and_every_start() {
    use crate::io::output::multi_start_yaml;
    assert_eq!(multi_start_yaml(&empty_fit_result()).unwrap(), "");

    let (best, _, starts) = select_start(mixed_results(), |r| r.ofv, false);
    let mut fit = best.unwrap().1;
    fit.n_starts = 6;
    fit.multi_start_seed = Some(42);
    fit.starts = starts;
    let ran = "n_iterations: 7, wall_time_secs: 0.250000";
    let want = [
        "".to_string(),
        "multi_start:".into(),
        "  n_starts: 6".into(),
        "  seed: 42".into(),
        "  selected_start: 3".into(),
        "  n_converged: 4".into(),
        "  n_not_converged: 1".into(),
        "  n_failed: 1".into(),
        "  compared: ofv".into(),
        "  best: 11.000000".into(),
        "  agree_within: 1.0".into(),
        "  n_compared: 3".into(),
        "  n_within: 3".into(),
        "  range: 1.000000".into(),
        "  starts:".into(),
        format!("    - {{start: 0, ofv: 12.000000, converged: true, status: converged, {ran}, selected: false, theta: [1.000000]}}"),
        "    - {start: 1, ofv: null, converged: false, status: failed, n_iterations: 0, wall_time_secs: 0.000000, selected: false, theta: [], error: \"start refused\"}".into(),
        format!("    - {{start: 2, ofv: 10.000000, converged: false, status: not_converged, {ran}, selected: false, theta: [2.000000]}}"),
        format!("    - {{start: 3, ofv: 11.000000, converged: true, status: converged, {ran}, selected: true, theta: [3.000000]}}"),
        format!("    - {{start: 4, ofv: null, converged: true, status: converged, {ran}, selected: false, theta: [4.000000]}}"),
        format!("    - {{start: 5, ofv: 11.000000, converged: true, status: converged, {ran}, selected: false, theta: [5.000000]}}"),
        "".into(),
    ]
    .join("\n");
    assert_eq!(multi_start_yaml(&fit).unwrap(), want);

    // Without a seed the key is not written.
    fit.multi_start_seed = None;
    assert!(!multi_start_yaml(&fit).unwrap().contains("seed:"));

    // Under a penalty the block says what it compared and each row carries the objective;
    // a coded failure carries its code.
    fit.starts[3].objective = Some(13.5);
    fit.starts[1].error_code = Some("E_X".into());
    let yaml = multi_start_yaml(&fit).unwrap();
    assert!(yaml.contains("  compared: penalized_objective\n"), "{yaml}");
    assert!(yaml.contains("  best: 13.500000\n"), "{yaml}");
    assert!(
        yaml.contains("{start: 3, ofv: 11.000000, objective: 13.500000, converged: true"),
        "{yaml}"
    );
    assert!(
        yaml.contains("error: \"start refused\", error_code: \"E_X\"}"),
        "{yaml}"
    );
}
