//! #2894: pins #2845's fix (#2921) on the model it was found on. `ode_method = auto` switches a
//! segment from an explicit stepper to a stiff one, and since #2921 never back. Before it, on
//! this full-TMDD model (30 simulated subjects, IV bolus 500, nine observations each; binding
//! runs at ~100/h right after the dose, then relaxes), `auto` demoted the stiff segment to RK45
//! on one probe reading and its objective moved off pinned `rodas4`.
//!
//! Evaluated at `maxiter = 0`, default tolerances. The regression is the single variable that
//! brings the demotion back: `let on_explicit = switching;` at the re-probe in
//! `integrate_dense_g` (`src/ode/solver.rs`), which also fails #2921's own unit test. Measured
//! on Linux x86_64 on `main` `f00221bf8`, `auto − rodas4` at the two points below, and `auto`
//! against a `vern7` reference at `ode_reltol = 1e-10`, `ode_abstol = 1e-13`:
//!
//! | TVKON | head `auto − rodas4` | demotion back | head `auto − ref` | demotion back |
//! |---|---|---|---|---|
//! | 1.326 | 6.7e-7 | −1.66e-3 | −0.0800 | −0.0817 |
//! | 1.340 | 3.4e-6 | −2.02e-3 | −0.0842 | −0.0862 |
//!
//! Over the 25 scan points (1.300 to 1.348, step 0.002) the demotion puts `auto − rodas4` at
//! 2.3e-4 to 5.3e-3. The head is at most 1e-5 from `rodas4` at ten of them, and the demotion is
//! at least 6.2e-4 off it at those ten, but not at the other fifteen: `auto` is not
//! bit-identical to `rodas4` here, because a segment that starts on RK45 may still switch to a
//! stiff method partway, and only a switch back is gone (head up to 3.9e-3 at 1.346). The two
//! points below are two of the ten. At them the bound of 1e-4 is 29× over the head's worse and
//! 16× under the demotion's smaller. The reference bound does not separate head from demotion:
//! the 0.08 is `rodas4`'s own default-tolerance error against `vern7`. It catches `auto` and
//! `rodas4` going wrong together by more than that, at 1.8× the head's gap; the references
//! are fixed here so the test does not run the slow tight solve.
//!
//! Scope: the threshold driver (`until_threshold_with`), the other site of the one-way gate, is
//! not reached, because this fixture has no event model.
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, FitOptions};
use std::path::Path;

const MODEL: &str = include_str!("fixtures/tmdd_auto_2845/model.ferx");
const DATA: &str = "tests/fixtures/tmdd_auto_2845/data.csv";

/// The objective at `TVKON = tvkon` (every other parameter at the file's values), no step.
fn ofv(method: &str, tvkon: f64) -> f64 {
    let src = MODEL.replace("  maxiter = 300", &format!("  maxiter = 0\n  ode_method = {method}"));
    assert!(src.contains("ode_method"), "fixture: fit options moved");
    let parsed = parse_full_model(&src).expect("the model parses");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("the data load");
    let mut params = parsed.model.default_params.clone();
    let k = parsed.model.theta_names.iter().position(|n| n == "TVKON").expect("TVKON");
    params.theta[k] = tvkon;
    let options =
        FitOptions { outer_maxiter: 0, run_covariance_step: false, ..parsed.fit_options.clone() }
            .quiet();
    fit(&parsed.model, &pop, &params, &options).expect("the evaluation runs").ofv
}

/// Four evaluations of a stiff 30-subject ODE model.
#[test]
fn auto_keeps_the_tmdd_segment_on_its_stiff_stepper() {
    for (tvkon, reference) in [(1.326, 1353.9916370867804), (1.340, 1420.1087978454166)] {
        let (auto, rodas4) = (ofv("auto", tvkon), ofv("rodas4", tvkon));
        assert!((auto - rodas4).abs() < 1e-4, "TVKON {tvkon}: auto {auto} vs rodas4 {rodas4}");
        assert!(
            (auto - reference).abs() < 0.15,
            "TVKON {tvkon}: auto {auto} vs the vern7 reference {reference}"
        );
    }
}
