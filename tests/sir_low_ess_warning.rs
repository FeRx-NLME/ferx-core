//! Tier 3 (#1723): SIR's low-ESS warning and `sir_scale = natural`, on fits to
//! convergence.
//!
//! * `warfarin_iov` fitted with **FOCEI** is the degenerate case: ESS 3.5 of
//!   1000, one draw holding 53% of the weight, and ETA_KA's variance on a
//!   likelihood shelf down to the box floor (conditional ΔOFV 1.56 there).
//! * `warfarin_iov` fitted with FOCE (ESS 405) and the `mbma_placebo` BTAV shape
//!   (ESS 143, `tests/data/mbma_placebo/`) are the healthy controls: no warning.
//! * Under `sir_scale = natural` the two swap: `warfarin_iov` FOCEI samples well
//!   (ESS 140) and `mbma_placebo` trips the warning (ESS 35), which then points
//!   back to `packed`.
//!
//! Every ESS literal below is the **Linux** value (AGENTS.md: Linux is the
//! reference platform for fit numbers), measured on Linux aarch64
//! (`tools/linux-test.sh`) and, for the three packed ones, identical in all
//! printed digits on the CI x86_64 `slow-tests.yml` dispatch. `ESS_REL_TOL` is
//! round-off headroom, not noise: the seed is fixed, so a changed ESS is a
//! changed computation. Native macOS arm64 stops its fits at slightly
//! different points (the OS libm, #1688) and differs by up to 1.04e-6 relative
//! (FOCEI packed: 3.5482961137825115), so these tests are red on macOS by
//! design; the regressions they exist for move the ESS by ≥ 12%.
//!
//! The per-PR tests of the same objects are in `estimation::sir::low_ess_tests`
//! (the message's input space, the probe's coordinates and floors, the
//! re-centre) and `estimation::uncertainty_samples::tests` (the Jacobian).

use ferx_core::types::{EstimationMethod, FitOptions, FitResult, SirScale, WarningCode};
use ferx_core::{fit, prepare_run, run_sir, PreparedRun};
use std::path::PathBuf;

const ESS_REL_TOL: f64 = 1e-9;

/// Print every ESS a test computes before any assertion can stop it, so one
/// run on a new platform reports all of them.
fn report(values: &[(&str, Option<f64>)]) {
    for (what, v) in values {
        eprintln!("MEASURED {what}: ESS {v:?}");
    }
}

fn assert_ess(got: Option<f64>, want: f64, what: &str) {
    let got = got.unwrap_or_else(|| panic!("{what}: no ESS"));
    assert!(
        got.is_finite() && ((got - want) / want).abs() < ESS_REL_TOL,
        "{what}: ESS {got}, measured {want}"
    );
}

fn sir_warnings(f: &FitResult) -> Vec<&String> {
    f.warnings.iter().filter(|w| w.starts_with("SIR")).collect()
}

/// The low-ESS warning reached `warnings_structured` too, classified `sir`.
fn assert_structured_sir(f: &FitResult) {
    assert!(
        f.warnings_structured.iter().any(|w| {
            w.category == WarningCode::Sir && w.message.starts_with("SIR: effective sample size")
        }),
        "{:?}",
        f.warnings_structured
    );
}

fn low_ess(f: &FitResult) -> Option<&String> {
    f.warnings
        .iter()
        .find(|w| w.starts_with("SIR: effective sample size"))
}

/// A fit with `sir = true` (1000 draws, 250 resamples, seed 1705) and the
/// covariance step, plus what `run_sir` needs to re-run SIR on it.
fn fit_with_sir(
    model: &str,
    data: &str,
    tweak: impl Fn(&mut FitOptions),
) -> (FitResult, FitOptions, PreparedRun) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let prep = prepare_run(
        root.join(model).to_str().unwrap(),
        Some(root.join(data).to_str().unwrap()),
    )
    .expect("prepare");
    let mut opts = FitOptions {
        verbose: false,
        run_covariance_step: true,
        sir: true,
        sir_samples: 1000,
        sir_resamples: 250,
        sir_seed: Some(1705),
        ..prep.parsed.fit_options.clone()
    };
    tweak(&mut opts);
    let f = fit(
        &prep.parsed.model,
        &prep.population,
        &prep.init_params,
        &opts,
    )
    .expect("fit");
    assert!(f.covariance_matrix.is_some(), "covariance step failed");
    (f, opts, prep)
}

fn rerun(f: &FitResult, opts: &FitOptions, prep: &PreparedRun, scale: SirScale) -> FitResult {
    let o = FitOptions {
        sir_scale: scale,
        ..opts.clone()
    };
    run_sir(f, Some(&prep.parsed.model), Some(&prep.population), &o).expect("run_sir")
}

/// S3 + the natural half on the same fit.
///
/// Packed (the default): exactly one low-ESS warning, ESS 3.5482966766833135 since #2093
/// (3.548442767315493 before it; 3.5482998071095064 before #1527, whose stationarity-certified fallback EBEs move this fit by ~4e-5 relative).
/// It names the heaviest draw's move (ETA_CL, −6.36 sd) and flags **only**
/// ETA_KA (ΔOFV 1.56): ETA_CL 9.31, ETA_V 18.09 and KAPPA_CL 76.78 are above
/// χ²₁(0.95) and must not be named in the shelf sentence. Mutations: drop the
/// warning push → `expect` dies; flag by the heaviest draw's coordinate instead
/// of the floor ΔOFV → "ETA_CL (ΔOFV" appears.
///
/// Natural, through `run_sir` on the fitted result: ESS 139.78477472131414 since #2093 (139.78747287250602; 139.78474460042662
/// before #1527) and
/// no low-ESS warning — the packed run's is replaced, not kept beside it. Dropping the re-centre
/// reads 57.3 (#1723 amendment A0), so the ESS assertion is the re-centre's
/// run-level kill.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn warfarin_iov_focei_warns_and_names_the_shelf_variance_and_natural_does_not() {
    let (f, opts, prep) =
        fit_with_sir("examples/warfarin_iov.ferx", "data/warfarin_iov.csv", |o| {
            o.method = EstimationMethod::FoceI;
            o.interaction = true;
        });
    // `run_sir` on the fitted result replaces the packed run's warning rather
    // than keeping it beside the new ESS.
    let nat = rerun(&f, &opts, &prep, SirScale::Natural);
    report(&[("FOCEI packed", f.sir_ess), ("FOCEI natural", nat.sir_ess)]);
    // #2039 moved both FOCEI literals (from 3.548_442_767_315_493 and
    // 139.787_472_872_506_02, 7.3e-7 and 3.6e-7 relative): a few inner trials with
    // V at 1e-93 or 1e169 overflow the closed form to a NaN amount, which the readout
    // used to serve as 0 and now serves as NaN, so those trials score differently.
    // Those NaN amounts are the only readouts that changed, and swapping
    // the old readout back in on one build moves the ESS by the same order (macOS
    // arm64: 3.548458 new, 3.548303 old). The new literals are the CI x86_64
    // `slow-tests.yml` value, identical in all printed digits in the core and nn jobs.
    // #2093 moved every ESS literal in this file: the fit reaches the same optimum (OFV equal
    // to six decimals on all three fits, worst θ 4.8e-6 relative, measured against `main`), but the
    // inner solver's step-scaled first iterate stops the EBEs at a slightly different point
    // and the fixed-seed SIR proposal is built from the covariance there.
    assert_ess(f.sir_ess, 3.548_296_676_683_313_5, "FOCEI packed");
    let w = low_ess(&f).unwrap_or_else(|| panic!("no low-ESS warning: {:?}", f.warnings));
    assert_eq!(
        f.warnings
            .iter()
            .filter(|x| x.starts_with("SIR: effective"))
            .count(),
        1,
        "{:?}",
        f.warnings
    );
    for s in [
        "effective sample size is 3.5 of 1000 draws",
        "it moves ETA_CL by -6.36 proposal standard deviations",
        "The data do not bound ETA_KA (ΔOFV 1.56 at variance",
        "`sir_scale = natural` makes these lower limits independent of the box.",
    ] {
        assert!(w.contains(s), "missing {s:?} in {w}");
    }
    for s in [
        "ETA_CL (ΔOFV",
        "ETA_V (ΔOFV",
        "KAPPA_CL (ΔOFV",
        "sir_scale = packed",
    ] {
        assert!(!w.contains(s), "unexpected {s:?} in {w}");
    }

    assert_structured_sir(&f);

    assert_ess(nat.sir_ess, 139.784_774_721_314_14, "FOCEI natural");
    assert!(low_ess(&nat).is_none(), "{:?}", nat.warnings);
}

/// S4: the healthy `warfarin_iov` FOCE fit (ESS 402.3820222439373 since #2217; 402.38567292526295 from #2202; 402.3856723316291 from #2093; 404.59 from #1722's frozen-variance FOCE EBEs; 396.21 before)
/// carries no SIR warning at all.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn warfarin_iov_foce_carries_no_sir_warning() {
    let (f, _, _) = fit_with_sir(
        "examples/warfarin_iov.ferx",
        "data/warfarin_iov.csv",
        |_| {},
    );
    report(&[("FOCE packed", f.sir_ess)]);
    // #2093 moved every ESS literal in this file: the fit reaches the same optimum (OFV equal
    // to six decimals on all three fits, worst θ 4.8e-6 relative, measured against `main`), but the
    // inner solver's step-scaled first iterate stops the EBEs at a slightly different point
    // and the fixed-seed SIR proposal is built from the covariance there.
    // #2202 moved it again, by 1.5e-9 relative (402.3856723316291 → 402.38567292526295, Linux
    // aarch64 and the x86_64 CI dispatch agree to every printed digit): the first outer
    // evaluation now warm-starts from the pre-flight EBEs, so the inner solves stop at a
    // slightly different point. The FOCEI literals above stayed inside `ESS_REL_TOL`.
    // #2217 moved it by −9.1e-6 relative (402.38567292526295 → 402.3820222439373; Linux
    // aarch64 and the x86_64 CI dispatch, run 39559756945, agree to every printed digit): the first outer evaluation now starts every occasion's κ at the pre-flight's
    // κ̂ instead of 0, so its inner solves stop at a different point inside `inner_tol`, the
    // outer path from there differs, and the fixed-seed SIR proposal is built from the
    // covariance at a slightly different point of the same optimum. Measured base against
    // head on one machine (macOS arm64, so for attribution only): the first OFV moves
    // 332.264851 → 332.264852, the optimizer stops at OFV 205.09032950834 → 205.09032950484
    // (−3.5e-9), worst θ TVKA 2.7e-6 relative, |g|∞ 1.5e-3 → 1.8e-4. The FOCEI literals
    // stayed inside `ESS_REL_TOL` again.
    // #2210 moved it by −2.3e-6 relative (402.3820222439373 → 402.38107815756786; Linux
    // x86_64 with CI's `ci-test` profile and `ci,slow-tests` features reproduces the old pin
    // to every printed digit at the base and the new value at `main` ebd20c6e, matching the
    // CI dispatches since): `div_taped` stopped recording a node for a constant zero over a
    // taped value, so the θ-gradient's reverse sweep sums in a different order. Its last bits
    // move the L-BFGS path, which stops at the same optimum: OFV 205.090329609855 →
    // 205.090329605773 (−2.0e-11 relative) after 44 → 48 evaluations, worst θ TVKA 5.5e-7
    // relative, measured base (fe129ea5) against #2210 (15041af3). The fixed-seed SIR
    // proposal is built from the covariance at that slightly different point. The FOCEI
    // literals above stayed inside `ESS_REL_TOL`.
    assert_ess(f.sir_ess, 402.381_078_157_567_86, "FOCE packed");
    assert!(sir_warnings(&f).is_empty(), "{:?}", f.warnings);
}

/// S4 + S10 on the MBMA BTAV shape, both halves in one test.
///
/// Packed (default): ESS 159.03342053931178 (143.22 before #2093), 1.59× the threshold, and no SIR
/// warning — the closest healthy fixture to the threshold, so moving it to
/// `sir_resamples` (250) reddens this half. Natural: ESS 43.09944242607953 (35.19 before #2093),
/// **with** the warning, which points back to `packed` and carries no box
/// sentence. Flipping the default to natural reddens the packed half.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn mbma_placebo_is_healthy_under_packed_and_warns_under_natural() {
    let (f, opts, prep) = fit_with_sir(
        "tests/data/mbma_placebo/mbma_placebo.ferx",
        "tests/data/mbma_placebo/mbma_placebo.csv",
        |_| {},
    );
    let nat = rerun(&f, &opts, &prep, SirScale::Natural);
    report(&[("MBMA packed", f.sir_ess), ("MBMA natural", nat.sir_ess)]);
    // #2093 moved every ESS literal in this file: the fit reaches the same optimum (OFV equal
    // to six decimals on all three fits, worst θ 4.8e-6 relative, measured against `main`), but the
    // inner solver's step-scaled first iterate stops the EBEs at a slightly different point
    // and the fixed-seed SIR proposal is built from the covariance there.
    assert_ess(f.sir_ess, 159.033_420_539_311_78, "MBMA packed");
    assert!(sir_warnings(&f).is_empty(), "{:?}", f.warnings);

    assert_ess(nat.sir_ess, 43.099_442_426_079_53, "MBMA natural");
    let w = low_ess(&nat).unwrap_or_else(|| panic!("no low-ESS warning: {:?}", nat.warnings));
    assert!(
        w.contains("`sir_scale = packed` may sample it better"),
        "{w}"
    );
    assert!(!w.contains("parameter box"), "{w}");
    // Through `run_sir`, whose result rebuilds the structured list.
    assert_structured_sir(&nat);
}
