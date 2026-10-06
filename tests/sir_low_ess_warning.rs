//! Tier 3 (#1723): SIR's low-ESS warning and `sir_scale = natural`, on fits to
//! convergence.
//!
//! * `warfarin_iov` fitted with **FOCEI** is the degenerate case: ESS 3.5 of
//!   1000, one draw holding 53% of the weight, and ETA_KA's variance on a
//!   likelihood shelf down to the box floor (conditional ΔOFV 1.56 there).
//! * `warfarin_iov` fitted with FOCE (ESS 396) and the `mbma_placebo` BTAV shape
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
/// Packed (the default): exactly one low-ESS warning, ESS 3.5482998071095064.
/// It names the heaviest draw's move (ETA_CL, −6.36 sd) and flags **only**
/// ETA_KA (ΔOFV 1.56): ETA_CL 9.31, ETA_V 18.09 and KAPPA_CL 76.78 are above
/// χ²₁(0.95) and must not be named in the shelf sentence. Mutations: drop the
/// warning push → `expect` dies; flag by the heaviest draw's coordinate instead
/// of the floor ΔOFV → "ETA_CL (ΔOFV" appears.
///
/// Natural, through `run_sir` on the fitted result: ESS 139.78474460042662 and
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
    assert_ess(f.sir_ess, 3.548_299_807_109_506_4, "FOCEI packed");
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

    assert_ess(nat.sir_ess, 139.784_744_600_426_62, "FOCEI natural");
    assert!(low_ess(&nat).is_none(), "{:?}", nat.warnings);
}

/// S4: the healthy `warfarin_iov` FOCE fit (ESS 396.21243679014765)
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
    assert_ess(f.sir_ess, 396.212_436_790_147_65, "FOCE packed");
    assert!(sir_warnings(&f).is_empty(), "{:?}", f.warnings);
}

/// S4 + S10 on the MBMA BTAV shape, both halves in one test.
///
/// Packed (default): ESS 143.22450068255318, 1.43× the threshold, and no SIR
/// warning — the closest healthy fixture to the threshold, so moving it to
/// `sir_resamples` (250) reddens this half. Natural: ESS 35.188868814200404,
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
    assert_ess(f.sir_ess, 143.224_500_682_553_18, "MBMA packed");
    assert!(sir_warnings(&f).is_empty(), "{:?}", f.warnings);

    assert_ess(nat.sir_ess, 35.188_868_814_200_404, "MBMA natural");
    let w = low_ess(&nat).unwrap_or_else(|| panic!("no low-ESS warning: {:?}", nat.warnings));
    assert!(
        w.contains("`sir_scale = packed` may sample it better"),
        "{w}"
    );
    assert!(!w.contains("parameter box"), "{w}");
    // Through `run_sir`, whose result rebuilds the structured list.
    assert_structured_sir(&nat);
}
