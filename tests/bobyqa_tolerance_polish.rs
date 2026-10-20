//! #1782: a BOBYQA tolerance stop is not a minimum.
//!
//! BOBYQA stops on `ftol_rel = 1e-6` / `xtol_rel = 1e-4`, both *relative*. At
//! |OFV| ≈ 430 a 4e-4 step meets the first while the trace is still falling, and
//! `FtolReached` used to map straight to `converged = true`. On `data/two_cpt_iv.csv`
//! (2-cmt IV, BLOCK(2) η on CL/V1, additive error, FOCEI) default BOBYQA reported
//! `Converged: YES` at −432.1645 on Linux and −432.0317 on macOS arm64, against the
//! −432.1668 L-BFGS, SLSQP and NONMEM all reach. The fix restarts BOBYQA from the best
//! point under tight tolerances after a tolerance stop, so the last leg's stop decides.
//!
//! The reference is NONMEM 7.6, `$EST METHOD=1 INTERACTION` on the same model and data
//! (see `tests/latched_creep_convergence.rs`, whose NONMEM run starts from the same
//! TVQ = 2): `MINIMIZATION SUCCESSFUL` at OFV −432.16683084.
//!
//! Tier 3: a fit to convergence (~1 s release).

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, Optimizer};
use std::path::Path;

/// The issue's `m_bobyqa.ferx`, without its `[fit_options]` (which `parse_model_string`
/// drops; the options are pinned on the `FitOptions` below instead).
const MODEL: &str = r"
[parameters]
  theta TVCL(4.0, 0.01, 100.0)
  theta TVV1(12.0, 0.1, 500.0)
  theta TVQ(2.0, 0.001, 100.0)
  theta TVV2(25.0, 0.1, 500.0)
  block_omega (ETA_CL, ETA_V1) = [0.1, 0.01, 0.1]
  sigma ADD_ERR ~ 1e-6
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)
[error_model]
  DV ~ additive(ADD_ERR)
";

const NONMEM_OFV: f64 = -432.166_830_842_677_9;

/// Before the fix the stop was 2.3e-3 (Linux x86_64, per the issue) and 0.135 (macOS
/// arm64, re-measured: −432.031734 with the polish disabled) above NONMEM. At the fix it
/// is −432.166831 on Linux arm64 (657 evals) and on macOS arm64 (1236 evals) — under
/// 1e-6 off. The 1e-3 bound sits below the smaller of the two false stops, so neither
/// platform's old answer passes it.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: fit to convergence (#1782); opt in with --features slow-tests"
)]
fn a_bobyqa_tolerance_stop_is_polished_to_the_optimum() {
    let model = parse_model_string(MODEL).expect("model parses");
    let pop = read_nonmem_csv(Path::new("data/two_cpt_iv.csv"), None, None).expect("data loads");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        optimizer: Optimizer::Bobyqa,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let r = fit(&model, &pop, &model.default_params, &opts).expect("fit runs");
    eprintln!(
        "bobyqa: converged {} OFV {:.6} evals {} theta {:?}",
        r.converged, r.ofv, r.n_iterations, r.theta
    );
    assert!(r.converged, "BOBYQA must converge");
    assert!(r.ofv.is_finite(), "OFV {}", r.ofv);
    assert!(
        (r.ofv - NONMEM_OFV).abs() < 1e-3,
        "OFV {} vs NONMEM {NONMEM_OFV}: a tolerance stop short of the optimum",
        r.ofv
    );
}
