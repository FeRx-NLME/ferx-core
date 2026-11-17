//! Tier-2 boundary tests for the opt-in PX-SAEM option `saem_px`.
//!
//! Each fit is a handful of iterations on the warfarin example: the tests pin the routing
//! (the option changes the iterates; the default does not; the result is shaped for the
//! original population), not convergence.

use ferx_core::parser::model_parser::{parse_full_model_file, parse_model_file};
use ferx_core::types::SaemPx;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

fn opts() -> FitOptions {
    let mut o = FitOptions::default();
    o.method = EstimationMethod::Saem;
    o.saem_n_exploration = Some(12);
    o.saem_n_convergence = 8;
    // The default burn-in (clamped to the 12 exploration iterations) would hold Omega fixed for the
    // whole exploration phase and PX, which acts on the Omega statistic, would have nothing to move.
    o.saem_omega_burnin = 2;
    o.saem_seed = Some(267);
    o.run_covariance_step = false;
    o.verbose = false;
    o.threads = Some(2);
    o
}

fn fit_with(o: &FitOptions) -> ferx_core::FitResult {
    let model = parse_model_file(Path::new("examples/warfarin.ferx")).expect("model");
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("data");
    fit(&model, &pop, &model.default_params, o).expect("SAEM fit")
}

/// `saem_px = off` is the plain algorithm: spelling the default out changes nothing, bit for bit.
#[test]
fn off_is_the_plain_algorithm() {
    let a = fit_with(&opts());
    let mut o = opts();
    o.saem_px = SaemPx::Off;
    let b = fit_with(&o);
    assert_eq!(a.ofv.to_bits(), b.ofv.to_bits());
}

/// Each mode moves the iterates (a switch that is not read would leave the OFV bit-identical
/// to the default) and still returns one EBE row per subject.
#[test]
fn each_mode_changes_the_iterates() {
    let base = fit_with(&opts());
    let n = base.subjects.len();
    for (label, mode) in [("all", SaemPx::All), ("exploration", SaemPx::Exploration)] {
        let mut o = opts();
        o.saem_px = mode;
        let r = fit_with(&o);
        assert_eq!(r.subjects.len(), n, "{label}: one EBE row per subject");
        assert!(r.ofv.is_finite(), "{label}: {}", r.ofv);
        assert_ne!(
            r.ofv.to_bits(),
            base.ofv.to_bits(),
            "{label} did not change the fit"
        );
    }
}

/// `exploration` applies the expanded update only while the exploration phase runs, `all`
/// also in the convergence phase: with a fixed 12 + 8 schedule the two fits differ.
#[test]
fn exploration_px_stops_with_the_exploration_phase() {
    let mut a = opts();
    a.saem_px = SaemPx::Exploration;
    let mut b = opts();
    b.saem_px = SaemPx::All;
    assert_ne!(
        fit_with(&a).ofv.to_bits(),
        fit_with(&b).ofv.to_bits(),
        "exploration and all must differ once a convergence phase runs"
    );
}

/// `[fit_options]` spellings reach the struct, and an unknown value is refused.
#[test]
fn fit_option_keys_parse() {
    let src = std::fs::read_to_string("examples/warfarin.ferx").expect("example");
    let dir = std::env::temp_dir().join(format!("ferx_saem_px_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let parse = |extra: &str| {
        let path = dir.join("m.ferx");
        std::fs::write(
            &path,
            src.replace("[fit_options]", &format!("[fit_options]\n  {extra}")),
        )
        .unwrap();
        parse_full_model_file(&path)
    };
    assert_eq!(
        parse("saem_px = all").expect("px").fit_options.saem_px,
        SaemPx::All
    );
    assert_eq!(
        parse("saem_px = exploration")
            .expect("px")
            .fit_options
            .saem_px,
        SaemPx::Exploration
    );
    assert_eq!(
        parse("saem_px = off").expect("px").fit_options.saem_px,
        SaemPx::Off
    );
    assert!(parse("saem_px = bogus").is_err());
    std::fs::remove_dir_all(&dir).ok();
}
