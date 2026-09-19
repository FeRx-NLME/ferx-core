//! Tier-3 convergence test for the SAEM MH step-scale adaptation rules (#1444).
//!
//! Run with:
//!
//!   cargo test --features slow-tests --test saem_scale_adaptation
//!
//! ## What this guards
//!
//! `ScaleAdaptation::Interval` — the default, and the only rule before #1444 —
//! multiplies the per-subject step scale `δ` by a fixed 1.1 or 0.9 every
//! `adapt_interval` iterations. Over the default 150 + 250 = 400 iterations at
//! `adapt_interval = 50` that is **8** corrections, so `δ` can move by at most
//! ≈2.1× up or ≈0.43× down *however far from target the chain is*.
//!
//! `examples/warfarin_saem.ferx` is a shipped model that needs far more than
//! that: from the 0.3 starting scale its combined block + componentwise
//! acceptance collapses to a few percent within ~25 iterations and stays there
//! for the whole run, against the 40% target.
//! `ScaleAdaptation::RobbinsMonro` steps `log δ` every iteration by
//! `c·k^-0.6·(accept_k − target)`, whose correction is proportional to the
//! discrepancy, and arrives.
//!
//! The assertion is a **differential pair that straddles the diagnostic's
//! gate**: the same model, the same seed, the same everything else, and the two
//! rules must land on opposite sides of `MH_RATE_LOW = 0.10`. The straddle
//! itself is asserted, so the pair cannot quietly become a tautology if the
//! model or the iteration counts are ever retuned.
//!
//! The observable is the per-iteration `mh_accept_rate` column of the optimizer
//! trace (`FitOptions::optimizer_trace`), which is the same combined block +
//! componentwise rate the end-of-run diagnostic folds — so this test measures
//! the quantity the warning reports, not a proxy for it.

use ferx_core::{fit, parse_model_file, read_nonmem_csv, EstimationMethod, FitOptions};

/// The `[fit_options] n_exploration + n_convergence` this test runs at.
const N_EXPLORE: usize = 150;
const N_CONVERGE: usize = 250;
/// `[fit_options] omega_burnin` default — iterations excluded from the tail.
const OMEGA_BURNIN: usize = 20;
/// The diagnostic's own lower band edge (`saem::MH_RATE_LOW`, `pub(crate)`).
const MH_RATE_LOW: f64 = 0.10;

/// Mean of the last `n` finite `mh_accept_rate` entries in a trace CSV.
///
/// Every row is checked with `is_finite()` before it is folded: a `NaN` from a
/// diverged solve would otherwise be swallowed by the accumulator and the mean
/// would describe only the rows that worked (CLAUDE.md's fold trap).
fn tail_accept_rate(trace_path: &str, n: usize) -> f64 {
    let txt = std::fs::read_to_string(trace_path)
        .unwrap_or_else(|e| panic!("cannot read trace {trace_path}: {e}"));
    let mut lines = txt.lines();
    let col = lines
        .next()
        .expect("trace has a header")
        .split(',')
        .position(|c| c == "mh_accept_rate")
        .expect("trace header has mh_accept_rate");
    let rates: Vec<f64> = lines
        .map(|l| {
            let f = l.split(',').nth(col).expect("row has the column");
            f.parse::<f64>()
                .unwrap_or_else(|e| panic!("unparseable mh_accept_rate {f:?}: {e}"))
        })
        .collect();
    assert!(
        rates.len() >= N_EXPLORE + N_CONVERGE,
        "expected {} SAEM rows, got {}",
        N_EXPLORE + N_CONVERGE,
        rates.len()
    );
    let tail = &rates[rates.len() - n..];
    let mut sum = 0.0;
    for &r in tail {
        assert!(r.is_finite(), "non-finite mh_accept_rate in {trace_path}");
        sum += r;
    }
    sum / tail.len() as f64
}

fn warfarin_tail_rate(rule: &str) -> f64 {
    let model = parse_model_file(std::path::Path::new("examples/warfarin_saem.ferx"))
        .expect("example parses");
    let pop = read_nonmem_csv(std::path::Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data loads");
    let mut opts = FitOptions {
        method: EstimationMethod::Saem,
        saem_n_exploration: N_EXPLORE,
        saem_n_convergence: N_CONVERGE,
        // The issue's measurement was taken at the *default* proposal count,
        // not the example file's `n_mh_steps = 3`.
        saem_n_mh_steps: 20,
        saem_omega_burnin: OMEGA_BURNIN,
        saem_seed: Some(12345),
        run_covariance_step: false,
        optimizer_trace: true,
        threads: Some(2),
        ..FitOptions::default()
    };
    // Via the parser, so the `[fit_options]` spelling is exercised too.
    ferx_core::parser::model_parser::apply_fit_option(&mut opts, "scale_adaptation", rule)
        .expect("scale_adaptation parses");
    let res = fit(&model, &pop, &model.default_params, &opts).expect("warfarin SAEM Ok");
    let path = res
        .trace_path
        .clone()
        .expect("optimizer_trace = true must produce a trace");
    let rate = tail_accept_rate(&path, 100);
    let _ = std::fs::remove_file(&path);
    println!("scale_adaptation = {rule}: tail acceptance {rate:.4}");
    rate
}

/// The defect of #1444 and its fix, as a straddle over the diagnostic's gate.
///
/// Realised on `origin/main @ 65118bea` + this change, seed 12345, 20 MH steps:
/// `interval` 0.0390, `robbins_monro` 0.4205 — matching the 0.040 / 0.420 the issue
/// reported from the research branch.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn warfarin_saem_reaches_target_only_under_robbins_monro() {
    let interval = warfarin_tail_rate("interval");
    let rm = warfarin_tail_rate("robbins_monro");
    assert!(
        interval.is_finite() && rm.is_finite(),
        "non-finite tail rates: interval {interval}, rm {rm}"
    );
    // The reported defect: the legacy rule never gets near 40%.
    assert!(
        interval < 0.08,
        "interval arm should still be stuck near a few percent, got {interval:.4}"
    );
    // The fix: Robbins-Monro arrives. A wide bracket — this is a stochastic
    // chain, and the claim is "reaches its target", not a pinned number.
    assert!(
        (0.30..=0.55).contains(&rm),
        "robbins_monro arm should reach the 40% target, got {rm:.4}"
    );
    // Assert the straddle itself. Without this the pair could be retuned into
    // two runs on the same side of the gate and still pass the two bounds above
    // if those were ever loosened.
    assert!(
        interval < MH_RATE_LOW && MH_RATE_LOW < rm,
        "the pair must straddle MH_RATE_LOW = {MH_RATE_LOW}: interval {interval:.4}, rm {rm:.4}"
    );
}
