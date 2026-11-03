//! Oracle O2 for nested variability levels (#1082 PR2): one explicit level is today's
//! inter-occasion variability, bit for bit.
//!
//! `level OCC by OCC in ID` with `kappa … at OCC` is the explicit spelling of
//! `iov_column = OCC`. It must fit to the same bits — OFV, θ, Ω, Ω_IOV and every κ̂ —
//! on the two IOV fixtures that carry a NONMEM anchor of their own: `warfarin_iov`
//! (unweighted) and `kappa_weight_iov` (a weight per arm, #1031). A `weight = 1` on the
//! unweighted kappa is the degenerate weight and must change nothing either. The `at`
//! tail is accepted on either side of a `weight =`.
//!
//! Each fit runs three outer iterations from the files' initial estimates, so the
//! comparison covers the inner EBE search, the outer gradient and an accepted step, not
//! only one objective evaluation.

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, FitOptions, FitResult};

fn run(model_src: &str, data: &str) -> FitResult {
    let parsed = parse_full_model(model_src).expect("model must parse");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        data,
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("data must load");
    let opts = FitOptions {
        outer_maxiter: 3,
        run_covariance_step: false,
        verbose: false,
        ..parsed.fit_options.clone()
    };
    fit(&parsed.model, &pop, &parsed.model.default_params, &opts).expect("fit")
}

/// `src` with its `iov_column = OCC` line replaced by an explicit level, and the kappa
/// line `kappa_line` replaced by `explicit`.
fn explicit(src: &str, kappa_line: &str, explicit: &str) -> String {
    assert!(
        src.contains("  iov_column = OCC\n"),
        "fixture has an iov_column line"
    );
    assert!(src.contains(kappa_line), "fixture has `{kappa_line}`");
    src.replace("  iov_column = OCC\n", "")
        .replace(kappa_line, &format!("level OCC by OCC in ID\n  {explicit}"))
}

fn bits(m: &nalgebra::DMatrix<f64>) -> Vec<u64> {
    m.iter().map(|v| v.to_bits()).collect()
}

fn assert_same_bits(label: &str, a: &FitResult, b: &FitResult) {
    assert!(a.ofv.is_finite(), "{label}: OFV {}", a.ofv);
    assert_eq!(
        a.ofv.to_bits(),
        b.ofv.to_bits(),
        "{label}: OFV {} vs {}",
        a.ofv,
        b.ofv
    );
    let tb = |r: &FitResult| r.theta.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(tb(a), tb(b), "{label}: θ");
    assert_eq!(bits(&a.omega), bits(&b.omega), "{label}: Ω");
    assert_eq!(
        a.omega_iov.as_ref().map(bits),
        b.omega_iov.as_ref().map(bits),
        "{label}: Ω_IOV"
    );
    assert_eq!(a.ebe_kappas.len(), b.ebe_kappas.len(), "{label}: subjects");
    for (i, (ka, kb)) in a.ebe_kappas.iter().zip(&b.ebe_kappas).enumerate() {
        let f = |k: &Vec<nalgebra::DVector<f64>>| {
            k.iter()
                .flat_map(|v| v.iter().map(|x| x.to_bits()))
                .collect::<Vec<_>>()
        };
        assert!(!ka.is_empty(), "{label}: subject {i} has κ̂");
        assert_eq!(f(ka), f(kb), "{label}: κ̂ of subject {i}");
    }
}

#[test]
fn one_explicit_level_is_iov_column_bit_for_bit_on_warfarin_iov() {
    let src = std::fs::read_to_string("examples/warfarin_iov.ferx").unwrap();
    let data = "data/warfarin_iov.csv";
    let today = run(&src, data);
    let line = "kappa KAPPA_CL ~ 0.01\n";
    for (label, tail) in [
        ("at OCC", "kappa KAPPA_CL ~ 0.01 at OCC\n"),
        (
            "at OCC weight = 1",
            "kappa KAPPA_CL ~ 0.01 at OCC weight = 1\n",
        ),
        (
            "weight = 1 at OCC",
            "kappa KAPPA_CL ~ 0.01 weight = 1 at OCC\n",
        ),
    ] {
        let model = explicit(&src, line, tail);
        assert_same_bits(label, &today, &run(&model, data));
    }
}

#[test]
fn one_explicit_level_is_iov_column_bit_for_bit_on_kappa_weight_iov() {
    let src = std::fs::read_to_string("tests/nonmem/kappa_weight_iov.ferx").unwrap();
    let data = "tests/nonmem/kappa_weight_iov.csv";
    let today = run(&src, data);
    assert!(
        today.kappa_weights.iter().any(|w| w.is_some()),
        "the fixture is weighted"
    );
    let line = "kappa KAPPA_CL ~ 0.01 weight = NARM\n";
    for (label, tail) in [
        (
            "at OCC weight = NARM",
            "kappa KAPPA_CL ~ 0.01 at OCC weight = NARM\n",
        ),
        (
            "weight = NARM at OCC",
            "kappa KAPPA_CL ~ 0.01 weight = NARM at OCC\n",
        ),
    ] {
        let model = explicit(&src, line, tail);
        let r = run(&model, data);
        assert_eq!(
            r.kappa_weights, today.kappa_weights,
            "{label}: the weight is kept"
        );
        assert_same_bits(label, &today, &r);
    }
}
