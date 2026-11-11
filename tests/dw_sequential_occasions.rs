//! #2829: the fit's Durbin–Watson statistic splits a subject's series only at
//! occasions that are **parallel**. Occasions that run in time (dosing intervals,
//! study periods) are one series again, as they were before #2826.
//!
//! Every fit here is an evaluation at the model's initial estimates
//! (`outer_maxiter = 0`, no covariance step), on the value path. The statistic is
//! post-fit arithmetic on the IWRES, so nothing is gained by converging. There is no
//! NONMEM column: NONMEM prints no Durbin–Watson statistic.
//!
//! Each test recomputes the statistic from the fit's own IWRES with
//! [`pair_normalised_dw`], written out here so the kernel is not graded with itself.
//!
//! ## Measured (macOS arm64, `c27f4723` + this change)
//!
//! | fixture | split at occasions | whole subject | the fit |
//! |---|---|---|---|
//! | sparse TDM, `iov_occasion = dose` | 0 pairs, `NaN` | 120 pairs, 0.8801 | whole |
//! | sparse TDM, `iov_column = OCC` | 0 pairs, `NaN` | 120 pairs, 0.8801 | whole |
//! | `warfarin_iov` (`iov_column = OCC`) | 200 pairs, 0.1293 | 210 pairs, 0.4177 | whole |
//! | naproxen + kappa, `iov_column = ARM` | 86 pairs, 2.3583 | 104 pairs, 2.1198 | split |
//!
//! The DW figures depend on η̂ and so on the platform in their last digits; the
//! assertions compare the fit against a recomputation on the same IWRES, which does
//! not.

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::stats::residual_error::iwres_autocorrelation;
use ferx_core::types::{classify_warning, WarningCode};
use ferx_core::{fit, FitOptions, FitResult, Population};

/// Lag pairs and pair-normalised DW of `series`.
fn pair_normalised_dw(series: &[Vec<f64>]) -> (usize, f64) {
    let (mut pairs, mut num, mut den) = (0usize, 0.0, 0.0);
    for s in series {
        for w in s.windows(2) {
            pairs += 1;
            num += (w[1] - w[0]).powi(2);
            den += (w[0] * w[0] + w[1] * w[1]) / 2.0;
        }
    }
    (pairs, num / den)
}

/// The fit's IWRES as one series per subject.
fn whole(r: &FitResult) -> Vec<Vec<f64>> {
    r.subjects.iter().map(|s| s.iwres.clone()).collect()
}

/// The fit's IWRES split at the caller's occasion labels (the `iov_column`'s).
fn by_occasion(pop: &Population, r: &FitResult) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for (s, res) in pop.subjects.iter().zip(&r.subjects) {
        assert_eq!(s.occasions.len(), res.iwres.len());
        let mut labels: Vec<u32> = Vec::new();
        for o in &s.occasions {
            if !labels.contains(o) {
                labels.push(*o);
            }
        }
        for label in labels {
            out.push(
                s.occasions
                    .iter()
                    .zip(&res.iwres)
                    .filter(|(o, _)| **o == label)
                    .map(|(_, e)| *e)
                    .collect(),
            );
        }
    }
    out
}

/// Subjects whose occasions run in time by the #2829 rule, recomputed here from the
/// public fields: each label one contiguous run, and a strictly later `obs_times`
/// at every label change.
fn subjects_in_time(pop: &Population) -> usize {
    pop.subjects
        .iter()
        .filter(|s| {
            let mut closed: Vec<u32> = Vec::new();
            (1..s.occasions.len()).all(|j| {
                if s.occasions[j] == s.occasions[j - 1] {
                    return true;
                }
                closed.push(s.occasions[j - 1]);
                s.obs_times[j] > s.obs_times[j - 1] && !closed.contains(&s.occasions[j])
            })
        })
        .count()
}

fn evaluate(src: &str, data: &str) -> (Population, FitResult) {
    let parsed = parse_full_model(src).expect("model parses");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        data,
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("data loads");
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let r = fit(&parsed.model, &pop, &parsed.model.default_params, &opts).expect("evaluation");
    assert!(
        r.subjects
            .iter()
            .all(|s| s.iwres.iter().all(|e| e.is_finite())),
        "every IWRES must be finite, or the hand recomputation pairs other rows"
    );
    (pop, r)
}

fn dw_warnings(r: &FitResult) -> Vec<&String> {
    r.warnings
        .iter()
        .filter(|w| classify_warning(w).category == WarningCode::DwAutocorrelation)
        .collect()
}

/// One-compartment oral model with IOV on clearance, for the sparse TDM data
/// (`tests/data/dw_sparse_tdm/`): six doses per subject, one trough per dosing
/// interval, AR(1) residuals.
const SPARSE: &str = "model sparse\n[parameters]\n  theta TVCL(0.2, 0.001, 10.0)\n  \
    theta TVV(10.0, 0.1, 500.0)\n  theta TVKA(1.5, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  \
    kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  \
    CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V  = TVV\n  KA = TVKA\n[structural_model]\n  \
    pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  DV ~ proportional(PROP_ERR)\n\
    [fit_options]\n  method = foce\n  covariance = false\n";

const SPARSE_DATA: &str = "tests/data/dw_sparse_tdm/sparse_tdm.csv";

/// T8, the exit condition of #2829. One observation per occasion: split at the
/// occasions there is no lag pair, the statistic was `NaN` and the warning silent.
/// Joined, the 24 subjects give 5 pairs each, the statistic is the whole-subject
/// one bit for bit, and the positive-autocorrelation warning the AR(1) residuals
/// call for is back. Under both ways of declaring the occasions: a derived rule and
/// a data column.
///
/// Mutation that must redden this: `occasions_run_in_time` always `false` (the gate
/// reverted; DW `NaN`, no warning).
#[test]
fn sparse_tdm_dw_is_taken_over_the_subject_and_warns() {
    for occasions in ["  iov_occasion = dose\n", "  iov_column = OCC\n"] {
        let src = format!("{SPARSE}{occasions}");
        let (pop, r) = evaluate(&src, SPARSE_DATA);
        assert_eq!(pop.subjects.len(), 24);
        assert!(
            pop.subjects
                .iter()
                .all(|s| s.observations.len() == 6 && s.doses.len() == 6),
            "{occasions}: six doses and six troughs per subject"
        );
        if occasions.contains("iov_column") {
            // The caller's labels are the column's: six occasions of one row each, in
            // time. Split there, no series has two rows.
            let (pairs, _) = pair_normalised_dw(&by_occasion(&pop, &r));
            assert_eq!(pairs, 0, "split at OCC there is no pair");
            assert_eq!(subjects_in_time(&pop), 24);
        }
        let (pairs, want) = pair_normalised_dw(&whole(&r));
        let (_, ungrouped) = iwres_autocorrelation(&r.subjects);
        eprintln!(
            "sparse TDM,{}: fit DW {:.10}, whole {want:.10} over {pairs} pairs",
            occasions.trim_end(),
            r.dw_statistic
        );
        assert_eq!(pairs, 120, "{occasions}: 24 subjects × 5 pairs");
        assert!(r.dw_statistic.is_finite(), "{occasions}: DW must be finite");
        assert_eq!(
            r.dw_statistic.to_bits(),
            ungrouped.to_bits(),
            "{occasions}: the fit's DW is the whole-subject one"
        );
        assert!((r.dw_statistic - want).abs() < 1e-12, "{occasions}: {want}");
        assert!(
            r.dw_statistic < 1.5,
            "{occasions}: AR(1) residuals, DW {}",
            r.dw_statistic
        );
        let dw = dw_warnings(&r);
        assert_eq!(dw.len(), 1, "{occasions}: {:#?}", r.warnings);
        assert!(
            dw[0].starts_with("Positive IWRES autocorrelation detected"),
            "{occasions}: {}",
            dw[0]
        );
    }
}

/// T9. `warfarin_iov`: two occasions per subject on one running clock, declared by
/// `iov_column = OCC`. The fit's DW is the whole-subject one (210 pairs), not the
/// per-occasion split (200 pairs), and the two differ by a wide margin, so the test
/// tells them apart.
///
/// Mutation that must redden this: `occasions_run_in_time` always `false`.
#[test]
fn warfarin_iov_dw_is_taken_over_the_subject() {
    let (pop, r) = evaluate(
        include_str!("../examples/warfarin_iov.ferx"),
        "data/warfarin_iov.csv",
    );
    assert_eq!(subjects_in_time(&pop), pop.subjects.len());
    assert!(
        pop.subjects
            .iter()
            .all(|s| s.occasions.windows(2).any(|w| w[0] != w[1])),
        "every subject must carry more than one occasion"
    );
    let (pairs_whole, want) = pair_normalised_dw(&whole(&r));
    let (pairs_split, split) = pair_normalised_dw(&by_occasion(&pop, &r));
    let (_, ungrouped) = iwres_autocorrelation(&r.subjects);
    eprintln!(
        "warfarin_iov: fit DW {:.10}, whole {want:.10} ({pairs_whole} pairs), \
         split by OCC {split:.10} ({pairs_split} pairs)",
        r.dw_statistic
    );
    assert_eq!((pairs_whole, pairs_split), (210, 200));
    assert!(r.dw_statistic.is_finite() && split.is_finite());
    assert_eq!(r.dw_statistic.to_bits(), ungrouped.to_bits());
    assert!((r.dw_statistic - want).abs() < 1e-12, "{want}");
    assert!(
        (r.dw_statistic - split).abs() > 0.1,
        "the fixture must tell the whole from the split: {} vs {split}",
        r.dw_statistic
    );
}

/// T10, the parallel control. The naproxen MBMA with a kappa on `E0` and
/// `iov_column = ARM`: every arm restarts the clock, so no study's occasions run in
/// time and the statistic stays split at the arms (86 pairs), away from the
/// whole-study value (104 pairs).
///
/// Mutation that must redden this: `occasions_run_in_time` always `true`.
#[test]
fn naproxen_arms_stay_split() {
    let src = include_str!("../examples/mbma_naproxen.ferx")
        .replace(
            "  omega ETA_EMAX ~ 0.763 (sd)\n",
            "  omega ETA_EMAX ~ 0.763 (sd)\n  kappa K_E0 ~ 0.3 (sd)\n",
        )
        .replace("+ ETA_E0\n", "+ ETA_E0 + K_E0\n")
        .replace("[fit_options]\n", "[fit_options]\n  iov_column = ARM\n");
    assert!(src.contains("kappa K_E0") && src.contains("ETA_E0 + K_E0"));
    let (pop, r) = evaluate(&src, "data/mbma_naproxen.csv");
    assert_eq!(pop.subjects.len(), 18);
    assert_eq!(subjects_in_time(&pop), 0, "every study's arms are parallel");
    let (pairs_whole, whole_dw) = pair_normalised_dw(&whole(&r));
    let (pairs_split, want) = pair_normalised_dw(&by_occasion(&pop, &r));
    eprintln!(
        "naproxen, iov_column = ARM: fit DW {:.10}, split by ARM {want:.10} ({pairs_split} \
         pairs), whole {whole_dw:.10} ({pairs_whole} pairs)",
        r.dw_statistic
    );
    assert_eq!((pairs_whole, pairs_split), (104, 86));
    assert!(r.dw_statistic.is_finite() && whole_dw.is_finite());
    assert!((r.dw_statistic - want).abs() < 1e-12, "{want}");
    assert!(
        (r.dw_statistic - whole_dw).abs() > 0.1,
        "the fixture must tell the split from the whole: {} vs {whole_dw}",
        r.dw_statistic
    );
}
