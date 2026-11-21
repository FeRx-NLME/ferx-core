//! #2894: pins #2845's fix (#2921) on the model it was found on. `ode_method = auto` switches a
//! segment from an explicit stepper to a stiff one, and since #2921 never back. Before it, on
//! this full-TMDD model (30 simulated subjects, IV bolus 500, nine observations each; binding
//! runs at ~100/h right after the dose, then relaxes), `auto` demoted the stiff segment to RK45
//! on one probe reading and its objective moved off pinned `rodas4` by up to 0.32 OFV.
//!
//! Evaluated at `maxiter = 0`, default tolerances, on #2845's scan points where the pre-#2921
//! build was furthest off. Measured (macOS arm64), `auto` against pinned `rodas4` and against a
//! `vern7` reference at `ode_reltol = 1e-10`, `ode_abstol = 1e-13`:
//!
//! | TVKON | main: `auto − rodas4` | before #2921 | main: `auto − ref` | before #2921 |
//! |---|---|---|---|---|
//! | 1.326 | 6.7e-7 | 0.320 | −0.080 | −0.398 |
//! | 1.340 | 3.4e-6 | 0.314 | −0.084 | −0.396 |
//!
//! Over all 25 scan points (1.300 to 1.348) the largest `|auto − rodas4|` is 5.2e-3 on main and
//! the smallest 0.113 before #2921. The bounds below sit between the two. `auto` is not
//! bit-identical to `rodas4` here, because a segment that starts on RK45 may still switch to
//! a stiff method partway, and only a switch back is gone. The reference bound also catches
//! `auto` and `rodas4` going wrong together. The references are the `vern7` objectives,
//! fixed here so the test does not run the slow tight solve.
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, FitOptions};
use std::path::Path;

const MODEL: &str = include_str!("fixtures/tmdd_auto_2845/model.ferx");
const DATA: &str = "tests/fixtures/tmdd_auto_2845/data.csv";

/// The objective at `TVKON = tvkon` (every other parameter at the file's values), no step.
fn ofv(method: &str, tvkon: f64) -> f64 {
    let src = MODEL.replace(
        "  maxiter = 300",
        &format!("  maxiter = 0\n  ode_method = {method}"),
    );
    assert!(src.contains("ode_method"), "fixture: fit options moved");
    let parsed = parse_full_model(&src).expect("the model parses");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("the data load");
    let mut params = parsed.model.default_params.clone();
    let k = parsed
        .model
        .theta_names
        .iter()
        .position(|n| n == "TVKON")
        .expect("TVKON");
    params.theta[k] = tvkon;
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    }
    .quiet();
    fit(&parsed.model, &pop, &params, &options)
        .expect("the evaluation runs")
        .ofv
}

/// Four evaluations of a stiff 30-subject ODE model: about 23 s in a debug build.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn auto_keeps_the_tmdd_segment_on_its_stiff_stepper() {
    for (tvkon, reference) in [(1.326, 1353.9916370867804), (1.340, 1420.1087978454166)] {
        let (auto, rodas4) = (ofv("auto", tvkon), ofv("rodas4", tvkon));
        assert!(
            (auto - rodas4).abs() < 1e-2,
            "TVKON {tvkon}: auto {auto} vs rodas4 {rodas4}"
        );
        assert!(
            (auto - reference).abs() < 0.15,
            "TVKON {tvkon}: auto {auto} vs the vern7 reference {reference}"
        );
    }
}
