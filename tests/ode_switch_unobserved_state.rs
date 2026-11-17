//! A moved `[odes]` switch in a state nothing observed reads is served by the exact gradient
//! (#2649): `examples/warfarin_ode_time.ferx` must fit without an FD fallback.
use ferx_core::{fit, parse_full_model_file, read_nonmem_csv};
use std::path::Path;

/// Regression caught: the moving-switch gate declining a switch that only feeds `TAM_INT`,
/// which routes every subject to finite-difference inner gradients.
#[test]
fn warfarin_ode_time_fits_without_an_fd_fallback() {
    let parsed = parse_full_model_file(Path::new("examples/warfarin_ode_time.ferx")).unwrap();
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).unwrap();
    let mut opts = parsed.fit_options;
    opts.outer_maxiter = 2;
    opts.run_covariance_step = false;
    let r = fit(&parsed.model, &pop, &parsed.model.default_params, &opts).expect("fit");
    assert!(r.ofv.is_finite(), "OFV {}", r.ofv);
    assert!(
        !r.warnings
            .iter()
            .any(|w| w.contains("finite-difference inner gradients")),
        "{:?}",
        r.warnings
    );
}
