//! NONMEM 7.5.1 anchor for the `power(σ, P)` residual-error form (#1182):
//! `Y = IPRED + IPRED**THETA(4)*EPS(1)`, FOCEI, on the warfarin dataset.
//!
//! Control streams and outputs are committed under `tests/nonmem/`:
//!
//! * `warfarin_power_eval.ctl` / `.lst` / `.ext` — `MAXEVAL=0` at the ferx
//!   initial estimates (`P = 1.3`, away from the proportional value where the
//!   exponent would be unreachable). This is the objective *function* anchor:
//!   both engines evaluate the same model at the same point.
//! * `warfarin_power.ctl` / `.lst` / `.ext` — the FOCEI fit. NONMEM
//!   terminated with `ROUNDING ERRORS (ERROR=134)`, as it commonly does on a
//!   power model whose σ and exponent trade off, at
//!   OBJ −286.7588 and `P = 1.112`; its estimates are the second evaluation
//!   point below, and the target the slow-gated ferx fit is compared with.
//!
//! | quantity | NONMEM 7.5.1 | ferx |
//! |---|---:|---:|
//! | OBJ at the initial estimates (`P = 1.3`) | 94.3818 | asserted to 5e-3 |
//! | OBJ at NONMEM's final estimates | −286.7588 | asserted to 5e-3 |
//! | fitted `P` (slow) | 1.1124 (SE 0.122) | asserted within 0.05 |
//! | fitted OBJ (slow) | −286.7588 | ferx must not be worse by more than 0.5 |

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const DATA: &str = "data/warfarin.csv";

/// `tests/nonmem/warfarin_power_eval.ext`, the `-1000000000` row.
const NM_EVAL_OBJ: f64 = 94.381804993706751;
/// `tests/nonmem/warfarin_power.ext`, the `-1000000000` row.
const NM_FIT_OBJ: f64 = -286.75882280193639;
const NM_TVCL: f64 = 0.132692;
const NM_TVV: f64 = 7.73957;
const NM_TVKA: f64 = 0.810868;
const NM_POW: f64 = 1.11244;
const NM_SIGMA2: f64 = 7.31043e-5;
const NM_OM_CL: f64 = 0.0285916;
const NM_OM_V: f64 = 0.00960805;
const NM_OM_KA: f64 = 0.335854;

fn model(theta: [f64; 4], omega: [f64; 3], sigma_sd: f64) -> ferx_core::types::CompiledModel {
    let src = format!(
        "[parameters]
  theta TVCL({}, 0.001, 10.0)
  theta TVV({}, 0.1, 500.0)
  theta TVKA({}, 0.01, 50.0)
  theta RUV_POW({}, 0.01, 10.0)
  omega ETA_CL ~ {}
  omega ETA_V  ~ {}
  omega ETA_KA ~ {}
  sigma PROP_ERR ~ {} (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ power(PROP_ERR, RUV_POW)

[fit_options]
  method = focei
",
        theta[0], theta[1], theta[2], theta[3], omega[0], omega[1], omega[2], sigma_sd
    );
    parse_model_string(&src).expect("model must parse")
}

fn evaluate(m: &ferx_core::types::CompiledModel) -> f64 {
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("data");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        checkpoint: false,
        threads: Some(2),
        ..FitOptions::default()
    };
    fit(m, &pop, &m.default_params, &opts)
        .expect("evaluation")
        .ofv
}

/// The objective function at the initial estimates: the same model, the same
/// point, two engines.
#[test]
fn power_objective_matches_nonmem_at_the_initial_estimates() {
    let m = model([0.13, 8.0, 1.0, 1.3], [0.09, 0.04, 0.30], 0.1);
    let ofv = evaluate(&m);
    eprintln!("power anchor, initial estimates: ferx {ofv:.6} vs NONMEM {NM_EVAL_OBJ:.6}");
    assert!(
        (ofv - NM_EVAL_OBJ).abs() < 5e-3,
        "ferx {ofv} vs NONMEM {NM_EVAL_OBJ}"
    );
}

/// The objective function at NONMEM's optimum — a second point, with the
/// exponent well away from one and σ two orders of magnitude smaller, so an
/// error in how the exponent enters the variance could not cancel across the
/// two points.
#[test]
fn power_objective_matches_nonmem_at_its_final_estimates() {
    let m = model(
        [NM_TVCL, NM_TVV, NM_TVKA, NM_POW],
        [NM_OM_CL, NM_OM_V, NM_OM_KA],
        NM_SIGMA2.sqrt(),
    );
    let ofv = evaluate(&m);
    eprintln!("power anchor, NONMEM optimum: ferx {ofv:.6} vs NONMEM {NM_FIT_OBJ:.6}");
    assert!(
        (ofv - NM_FIT_OBJ).abs() < 5e-3,
        "ferx {ofv} vs NONMEM {NM_FIT_OBJ}"
    );
}

/// The full FOCEI fit lands where NONMEM's did.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn power_fit_matches_nonmem() {
    let m = model([0.13, 8.0, 1.0, 1.3], [0.09, 0.04, 0.30], 0.1);
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("data");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        run_covariance_step: false,
        verbose: false,
        checkpoint: false,
        ..FitOptions::default()
    };
    let r = fit(&m, &pop, &m.default_params, &opts).expect("fit");
    assert!(
        r.ofv <= NM_FIT_OBJ + 0.5,
        "ferx OFV {} is worse than NONMEM's {NM_FIT_OBJ}",
        r.ofv
    );
    let p = r.theta[3];
    assert!((p - NM_POW).abs() < 0.05, "exponent {p} vs NONMEM {NM_POW}");
    assert!(
        (r.theta[0] - NM_TVCL).abs() / NM_TVCL < 0.05,
        "TVCL {}",
        r.theta[0]
    );
    assert!(
        (r.theta[1] - NM_TVV).abs() / NM_TVV < 0.05,
        "TVV {}",
        r.theta[1]
    );
}
