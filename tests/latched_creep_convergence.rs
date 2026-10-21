//! #1934: the stagnation guard must not latch a fit that is still descending.
//!
//! The short window (#1530) latches after `max(n+1, 10)` evals without a 1e-3 OFV
//! improvement, and the plateau check that then decides `converged` tests only whether
//! the OFV stopped moving. On `data/two_cpt_iv.csv` with the start TVQ = 0.01, SLSQP
//! creeps along a valley in which it gains less than 1e-3 per ten evals: before the fix
//! the guard latched at eval 60 and the fit reported `Converged: YES` at OFV −204.4111,
//! 227.76 short of the optimum. The fix asks the BHHH Newton decrement at the incumbent
//! before latching (predicted gain 14.8 there), so the fit keeps going.
//!
//! The reference is NONMEM 7.6, `$EST METHOD=1 INTERACTION` on the same model and data
//! (ADVAN3 TRANS4, `$OMEGA BLOCK(2)`, additive `$SIGMA`): from TVQ = 2 it reports
//! `MINIMIZATION SUCCESSFUL` at OFV −432.16683084; the estimates below are its `.ext`
//! final row. From TVQ = 0.01 NONMEM stalls too, at −204.4144 near TVQ 0.0112, and
//! terminates on MAXEVAL with "NO. OF SIG. DIGITS UNREPORTABLE" — the same valley, which
//! NONMEM at least reports as a failure.
//!
//! Tier 3: a fit to convergence (~0.1 s release).

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, Optimizer};
use std::path::Path;

const MODEL_Q_001: &str = r"
[parameters]
  theta TVCL(4, 0.01, 100.0)
  theta TVV1(12, 0.1, 500.0)
  theta TVQ(0.01, 0.0001, 100.0)
  theta TVV2(25, 0.1, 5000.0)
  block_omega (ETA_CL, ETA_V1) = [0.1, 0.01, 0.1]
  sigma ADD_ERR ~ 0.01
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

/// NONMEM's OFV and θ (TVCL, TVV1, TVQ, TVV2) at its FOCEI optimum.
const NONMEM_OFV: f64 = -432.166_830_842_677_9;
const NONMEM_THETA: [f64; 4] = [4.80976, 15.4816, 2.94425, 19.0347];

/// [`MODEL_Q_001`] as an `[odes]` model solved by `ode_method` (#2001). The system is
/// linear, so `auto` solves it exactly by matrix exponential (#852) and `rk45` integrates it.
fn ode_model_q_001(ode_method: &str) -> String {
    MODEL_Q_001.replace(
        "  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)\n",
        &format!(
            "  ode(states=[central, peripheral])
[odes]
  d/dt(central)    = -(CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central  - (Q/V2) * peripheral
[scaling]
  y = central / V1
[fit_options]
  ode_method = {ode_method}
"
        ),
    )
}

fn fit_from_q_001(optimizer: Optimizer) -> ferx_core::FitResult {
    fit_source_from_q_001(MODEL_Q_001, optimizer)
}

fn fit_source_from_q_001(source: &str, optimizer: Optimizer) -> ferx_core::FitResult {
    let model = parse_model_string(source).expect("model parses");
    let pop = read_nonmem_csv(Path::new("data/two_cpt_iv.csv"), None, None).expect("data loads");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        optimizer,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    fit(&model, &pop, &model.default_params, &opts).expect("fit runs")
}

/// Measured on Linux x86_64 at the fix: SLSQP ends at −432.166694 (1.4e-4 above
/// NONMEM) with every θ within 1.3e-4 relative of NONMEM's; L-BFGS at −432.166831
/// (≤ 1e-6 off). The bounds are 1e-2 OFV and 2e-3 relative — 70× and 15× headroom on
/// the worse of the two — and the false stop they replace was 227.76 OFV and 99.6% in
/// TVQ away, so neither bound is anywhere near passing it.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: fit to convergence (#1934); opt in with --features slow-tests"
)]
fn a_slow_creep_is_not_latched_as_converged() {
    // L-BFGS from the same start is the control: it never crept, so it reached the
    // optimum before #1934 and must still. SLSQP is the case the fix is for.
    for optimizer in [Optimizer::Slsqp, Optimizer::NloptLbfgs] {
        let r = fit_from_q_001(optimizer);
        eprintln!(
            "{optimizer:?}: converged {} OFV {:.6} theta {:?}",
            r.converged, r.ofv, r.theta
        );
        assert!(r.converged, "{optimizer:?} must converge");
        assert!(r.ofv.is_finite(), "{optimizer:?}: OFV {}", r.ofv);
        assert!(
            (r.ofv - NONMEM_OFV).abs() < 1e-2,
            "{optimizer:?}: OFV {} vs NONMEM {NONMEM_OFV}",
            r.ofv
        );
        for (k, (&got, &want)) in r.theta.iter().zip(&NONMEM_THETA).enumerate() {
            assert!(got.is_finite(), "{optimizer:?} theta[{k}] = {got}");
            assert!(
                ((got - want) / want).abs() < 2e-3,
                "{optimizer:?} theta[{k}]: {got} vs NONMEM {want}"
            );
        }
    }
}

/// The warning a stagnation latch adds when it overrode the decrement veto.
const VETO_OVERRIDE: &str = "Stagnation guard stopped the outer optimizer although";

/// #2001: the same creep on the model's ODE twin. #1999 withheld the decrement from every
/// ODE model, so both runs below latched silently: SLSQP on the exact (matrix-exponential)
/// path at eval 66 on −204.4130, and L-BFGS under rk45 at eval 27 on −204.4227, each
/// reported converged 227.7 OFV short of NONMEM's optimum with no warning.
///
/// Measured on Linux x86_64 at the fix. SLSQP on the exact path now reaches −432.166537,
/// 2.9e-4 above NONMEM, every θ within 5.4e-4 relative; the bounds are those of the
/// closed-form test above (34× and 3.7× headroom). L-BFGS under
/// rk45 is vetoed at eval 27 (predicted gain 14.83, the closed form's at the same point)
/// but does not escape: on the default-tolerance integrated gradient its line search makes
/// no progress, and with the guard off NLopt stops at the same point itself
/// (`XtolReached`, −204.42); at `ode_reltol = 1e-9` it reaches −432.1668. So the guard
/// stops it after the re-armed window and says so, which is all it can do for a fit the
/// optimizer has itself abandoned. Silence is the regression this pins.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: fit to convergence (#2001); opt in with --features slow-tests"
)]
fn a_slow_creep_on_an_exact_ode_model_reaches_the_optimum() {
    let exact = fit_source_from_q_001(&ode_model_q_001("auto"), Optimizer::Slsqp);
    eprintln!(
        "exact SLSQP: converged {} OFV {:.6} theta {:?}",
        exact.converged, exact.ofv, exact.theta
    );
    assert!(exact.converged, "exact SLSQP must converge");
    assert!(
        exact.ofv.is_finite() && (exact.ofv - NONMEM_OFV).abs() < 1e-2,
        "exact SLSQP: OFV {} vs NONMEM {NONMEM_OFV}",
        exact.ofv
    );
    for (k, (&got, &want)) in exact.theta.iter().zip(&NONMEM_THETA).enumerate() {
        assert!(
            got.is_finite() && ((got - want) / want).abs() < 2e-3,
            "exact SLSQP theta[{k}]: {got} vs NONMEM {want}"
        );
    }
}

/// The rk45 L-BFGS half of [`a_slow_creep_on_an_exact_ode_model_reaches_the_optimum`], its
/// own test so each side's regression names itself.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: fit to convergence (#2001); opt in with --features slow-tests"
)]
fn a_slow_creep_on_an_integrated_ode_model_is_not_latched_silently() {
    let rk45 = fit_source_from_q_001(&ode_model_q_001("rk45"), Optimizer::NloptLbfgs);
    eprintln!(
        "rk45 L-BFGS: converged {} OFV {:.6} warnings {:?}",
        rk45.converged, rk45.ofv, rk45.warnings
    );
    assert!(rk45.ofv.is_finite(), "rk45 L-BFGS: OFV {}", rk45.ofv);
    let short = rk45.ofv - NONMEM_OFV;
    let warned = rk45.warnings.iter().any(|w| w.contains(VETO_OVERRIDE));
    assert!(
        short < 1e-2 || warned || !rk45.converged,
        "rk45 L-BFGS stopped {short} short of NONMEM, reported converged, with no warning"
    );
}
