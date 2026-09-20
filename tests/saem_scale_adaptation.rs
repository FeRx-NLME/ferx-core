//! Tier-3 convergence test for the SAEM MH step-scale adaptation rules (#1444).
//!
//! Run with:
//!
//!   cargo test --features slow-tests --test saem_scale_adaptation
//!
//! ## What this guards
//!
//! `ScaleAdaptation::Interval` — the only rule before #1444, and the default
//! until #1449 — multiplies the per-subject step scale `δ` by a fixed 1.1 or 0.9 every
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
//! The observable is `FitResult::saem_mh_accept_tail`, the combined block +
//! componentwise acceptance over the trailing post-burn-in iterations — the
//! same number the end-of-run diagnostic reports on, so this test measures the
//! quantity the warning states rather than a proxy for it.

use ferx_core::{fit, parse_model_file, read_nonmem_csv, EstimationMethod, FitOptions};

/// The `[fit_options] n_exploration + n_convergence` this test runs at.
const N_EXPLORE: usize = 150;
const N_CONVERGE: usize = 250;
/// `[fit_options] omega_burnin` default — iterations excluded from the tail.
const OMEGA_BURNIN: usize = 20;
/// The diagnostic's own lower band edge (`saem::MH_RATE_LOW`, `pub(crate)`).
const MH_RATE_LOW: f64 = 0.10;

/// The rule under test with the **dead band switched off**, so that the pair
/// below measures the scale rules themselves. Since #1449 the shipped default
/// carries `scale_deadband = 0.15,0.60`, and leaving that in would make the
/// `robbins_monro` arm a *banded* arm — a different object, measured by
/// `the_default_deadband_still_rescues_warfarin`.
fn warfarin_tail_rate(rule: &str) -> f64 {
    warfarin_tail_rate_seed(rule, 12345)
}

fn warfarin_tail_rate_seed(rule: &str, seed: u64) -> f64 {
    warfarin_tail_rate_banded(rule, seed, Some("none"))
}

fn warfarin_tail_rate_banded(rule: &str, seed: u64, band: Option<&str>) -> f64 {
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
        saem_seed: Some(seed),
        run_covariance_step: false,
        threads: Some(2),
        ..FitOptions::default()
    };
    // Via the parser, so the `[fit_options]` spelling is exercised too.
    ferx_core::parser::model_parser::apply_fit_option(&mut opts, "scale_adaptation", rule)
        .expect("scale_adaptation parses");
    if let Some(b) = band {
        ferx_core::parser::model_parser::apply_fit_option(&mut opts, "scale_deadband", b)
            .expect("scale_deadband parses");
    }
    let res = fit(&model, &pop, &model.default_params, &opts).expect("warfarin SAEM Ok");
    let rate = res
        .saem_mh_accept_tail
        .expect("a 400-iteration SAEM fit must report a tail acceptance");
    // A `NaN` from a diverged solve would sail through every comparison below
    // (any comparison against NaN is false, so a `<` bound would simply fail,
    // but a range `contains` would too) — assert finiteness explicitly so the
    // failure names the real problem.
    assert!(rate.is_finite(), "non-finite tail acceptance: {rate}");
    println!("scale_adaptation = {rule}: tail acceptance {rate:.4}");
    rate
}

/// The defect of #1444 and its fix, as a straddle over the diagnostic's gate.
///
/// **The bounds below are measured, not bracketed** (#1451 review). Realised tail
/// acceptance on this machine, `origin/main @ 65118bea` + this change, 20 MH steps,
/// over a four-seed sweep:
///
/// | seed  | `interval` | `robbins_monro` |
/// |-------|-----------:|----------------:|
/// | 1     |     0.0396 |          0.4228 |
/// | 2     |     0.0408 |          0.4178 |
/// | 777   |     0.0412 |          0.4170 |
/// | 12345 |     0.0390 |          0.4205 |
///
/// so the worst realised values are `interval` **0.0412** (cross-seed spread
/// 0.0022) and `|robbins_monro − 0.40|` **0.0228** (spread 0.0058). The bounds are
/// derived from those with stated headroom: `interval < 0.06` is 1.46x the worst
/// observed value and still well below the diagnostic's own `MH_RATE_LOW = 0.10`,
/// and `|rm − 0.40| < 0.08` is 3.5x the worst observed deviation. The seed-12345
/// pair the test itself runs (0.0390 / 0.4205) reproduces the 0.040 / 0.420 the
/// issue reported from the research branch.
///
/// Re-measured on this commit (#1449), with the dead band explicitly off so
/// that this pair still measures the two *rules*: 0.0394 / 0.4201 at seed
/// 12345 — both inside the bounds above, and both moved by less than the
/// cross-seed spread, so the bounds were not recalibrated.
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
    // The reported defect: the legacy rule never gets near 40%. Bound from the
    // sweep's worst 0.0412, with 1.46x headroom.
    assert!(
        interval < 0.06,
        "interval arm should still be stuck near a few percent, got {interval:.4}"
    );
    // The fix: Robbins-Monro arrives. Bound from the sweep's worst deviation
    // 0.0228, with 3.5x headroom.
    assert!(
        (rm - 0.40_f64).abs() < 0.08,
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

/// The dead band of #1449 must not cost warfarin its rescue.
///
/// The band exists so that a chain already *near* target is left alone; the
/// whole point of #1444's Robbins-Monro rule is a chain that is nowhere near
/// it. Warfarin is that chain — a few percent against a 40% target — so the
/// **shipped default**, which carries the band, must still rescue it: the
/// banded arm has to land on the same side of `MH_RATE_LOW` as the unbanded
/// one while the legacy rule stays on the other side.
///
/// The band's own effect is bounded rather than asserted to be nil, because it
/// is **not** nil: with `n_mh_steps = 20` a subject's per-iteration rate is a
/// multiple of 0.05, so once the chain has arrived many iterations fall inside
/// `[0.15, 0.60]` and are skipped, and the censored steps that remain do not
/// cancel at the target. Realised tail acceptance, seed 12345, 20 MH steps,
/// measured on this commit: `interval` **0.0394**, `robbins_monro` unbanded
/// **0.4201**, `robbins_monro` + the default band **0.3396**. So the band
/// costs 0.08 of acceptance here — an eighth of the way back to the legacy
/// rule's 0.04 — and the bound below is `|rate − 0.40| < 0.12`, i.e. twice the
/// realised 0.0604 deviation. A looser bound than the unbanded arm's 0.08 is
/// not a weaker test of the same thing: it is the bound for a different,
/// measured quantity, and the straddle assertion is what pins the rescue.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn the_default_deadband_still_rescues_warfarin() {
    let interval = warfarin_tail_rate("interval");
    let banded = warfarin_tail_rate_banded("robbins_monro", 12345, Some("0.15,0.60"));
    assert!(
        interval.is_finite() && banded.is_finite(),
        "non-finite tail rates: interval {interval}, banded {banded}"
    );
    assert!(
        (banded - 0.40_f64).abs() < 0.12,
        "the banded arm must still reach the 40% target, got {banded:.4}"
    );
    assert!(
        interval < MH_RATE_LOW && MH_RATE_LOW < banded,
        "the pair must straddle MH_RATE_LOW = {MH_RATE_LOW}: \
         interval {interval:.4}, banded {banded:.4}"
    );
}
