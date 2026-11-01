//! #2544 on the shipped `tests/data/mbma_placebo` model: the one-line
//! `BASE = TVE0 + PLACEBO + ETA_E0 + KAPPA_ARM` against the split spelling
//! `E0 = TVE0 + ETA_E0`, `BASE = E0 + PLACEBO + KAPPA_ARM`. Both now carry the same
//! `ETA_E0 → TVE0` additive mu-reference.
//!
//! - FOCEI (T6). The FOCE inner loop optimises in `ψ = η + μ`, so a new mu-reference
//!   changes its coordinates and the one-line spelling's objective is no longer the
//!   byte-for-byte value it had before #2544. It now runs the split spelling's
//!   coordinates; the two agree to a measured bound, not bit for bit, since the sums
//!   associate differently. The bound is a guard that the FOCE change stays below the
//!   printed digits, not a detector of the mu-reference: FOCEI's optimum does not depend
//!   on the centring, and with the one-line arm disabled the gap is the same 1.3e-12.
//!   That is the plan's proposed mutation (μ = 0 for one spelling), so no mutation of this
//!   PR's code reddens the bound; it would take an FOCEI regression that makes the two
//!   spellings reach different optima. What dies when the arm goes is the
//!   `ETA_E0 → TVE0` assertion in `both`.
//! - SAEM (T7, `slow-tests`). Per seed the spellings differ (the summation order changes
//!   the floating-point path and the chain diverges), so the check is distributional:
//!   five seeds of each, the one-line median within the split's runs and every run near
//!   the FOCEI minimum (97.070036). Before #2544 the one-line spelling stopped 1–11 OFV
//!   units above it (100.97–107.92 on these seeds, macOS arm64).

use std::path::{Path, PathBuf};

const ONE_LINE: &str = "  BASE = TVE0 + PLACEBO + ETA_E0 + KAPPA_ARM\n";
const SPLIT: &str = "  E0   = TVE0 + ETA_E0\n  BASE = E0 + PLACEBO + KAPPA_ARM\n";

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/mbma_placebo")
}

/// The shipped model, with `[individual_parameters]` spelled `indiv`, written next to
/// a copy of the data in a fresh temporary directory.
fn spelled(indiv: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let text = std::fs::read_to_string(data_dir().join("mbma_placebo.ferx")).unwrap();
    assert!(
        text.contains(ONE_LINE),
        "the shipped model is the one-line spelling"
    );
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("m.ferx");
    let data = dir.path().join("d.csv");
    std::fs::write(&model, text.replace(ONE_LINE, indiv)).unwrap();
    std::fs::copy(data_dir().join("mbma_placebo.csv"), &data).unwrap();
    (dir, model, data)
}

fn fit(
    model: &Path,
    data: &Path,
    tweak: impl Fn(&mut ferx_core::types::FitOptions),
) -> ferx_core::types::FitResult {
    let prep = ferx_core::prepare_run(model.to_str().unwrap(), Some(data.to_str().unwrap()))
        .expect("prepare");
    let mut opts = ferx_core::types::FitOptions {
        verbose: false,
        run_covariance_step: false,
        ..prep.parsed.fit_options.clone()
    };
    tweak(&mut opts);
    ferx_core::fit(
        &prep.parsed.model,
        &prep.population,
        &prep.init_params,
        &opts,
    )
    .expect("fit")
}

fn both(
    tweak: impl Fn(&mut ferx_core::types::FitOptions) + Copy,
) -> [ferx_core::types::FitResult; 2] {
    [ONE_LINE, SPLIT].map(|indiv| {
        let (_dir, model, data) = spelled(indiv);
        let prep = ferx_core::prepare_run(model.to_str().unwrap(), Some(data.to_str().unwrap()))
            .expect("prepare");
        assert_eq!(
            prep.parsed
                .model
                .mu_refs
                .get("ETA_E0")
                .map(|r| r.theta_name.as_str()),
            Some("TVE0"),
            "{indiv}: ETA_E0 → TVE0"
        );
        fit(&model, &data, tweak)
    })
}

#[test]
fn focei_on_the_two_spellings_agrees() {
    let [one, split] = both(|o| o.method = ferx_core::types::EstimationMethod::FoceI);
    let theta_gap = one
        .theta
        .iter()
        .zip(&split.theta)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0f64, |m, g| {
            assert!(g.is_finite(), "theta gap {g}");
            m.max(g)
        });
    let ofv_gap = (one.ofv - split.ofv).abs();
    assert!(ofv_gap.is_finite());
    eprintln!(
        "mbma_placebo FOCEI: one-line OFV {:.9}, split {:.9}, |gap| {ofv_gap:.3e}, worst \
         relative theta gap {theta_gap:.3e}",
        one.ofv, split.ofv
    );
    assert!(ofv_gap < OFV_BOUND, "OFV gap {ofv_gap:e}");
    assert!(theta_gap < THETA_BOUND, "theta gap {theta_gap:e}");
}

// Measured |OFV gap| 1.322e-12 and worst relative θ gap 5.707e-8 on Linux aarch64
// (1.364e-12 / 5.710e-8 on macOS arm64); the bounds are 75× and 17× that.
const OFV_BOUND: f64 = 1e-10;
const THETA_BOUND: f64 = 1e-6;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn saem_on_the_one_line_spelling_lands_with_the_split_one() {
    let seeds = [12345u64, 2, 3, 7, 11];
    let ofvs = |indiv: &str| -> Vec<f64> {
        let (_dir, model, data) = spelled(indiv);
        seeds
            .iter()
            .map(|&s| {
                let f = fit(&model, &data, |o| {
                    o.method = ferx_core::types::EstimationMethod::Saem;
                    o.saem_seed = Some(s);
                });
                assert!(f.ofv.is_finite(), "{indiv} seed {s}: {}", f.ofv);
                f.ofv
            })
            .collect()
    };
    let median = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        s[s.len() / 2]
    };
    let (one, split) = (ofvs(ONE_LINE), ofvs(SPLIT));
    eprintln!("mbma_placebo SAEM seeds {seeds:?}: one-line {one:?}, split {split:?}");
    // Five-seed medians are too noisy to compare directly (the gap was 0.23 on Linux
    // aarch64, 0.05 on macOS arm64, and over 20 seeds the two medians are 98.31 / 98.33),
    // so the one-line median must sit within the split spelling's own runs.
    let split_worst = split.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        median(&one) <= split_worst,
        "one-line median {} above the split's worst {split_worst}: one-line {one:?}, \
         split {split:?}",
        median(&one)
    );
    for (s, o) in seeds.iter().zip(&one) {
        assert!(*o < WORST_BOUND, "one-line seed {s}: {o}");
    }
}

// Linux aarch64: one-line 97.28–97.85, split 97.34–98.05; macOS arm64: one-line
// 97.21–98.88, split 97.33–98.39. 99.5 is 1.65 above the Linux worst and 0.6 above the
// macOS one; before #2544 all five seeds were above it (100.97–107.92, macOS).
const WORST_BOUND: f64 = 99.5;
