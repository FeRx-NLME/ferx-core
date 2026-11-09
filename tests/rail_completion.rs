//! Rail completion (#2832): a variance BOBYQA stopped short of its internal lower guard is
//! resumed from just above the guard.
//!
//! `tests/data/guard_hold/mix_override_collapse.{py,csv}` (#2809): two latent classes of 30
//! subjects, IIV on CL in class 1 only. A `[mixture]` model estimating a class-2 override
//! `omega(2) ETA_CL` drives it toward zero; the objective falls monotonically to the guard
//! with a geometrically flattening tail, and BOBYQA's `xtol_rel` stop lands short of it
//! (Linux aarch64 at `7c63c7a5d`: packed −4.8045, OFV −37.238760). The reference optimum,
//! with the override on the guard, is −37.397489713, reached by L-BFGS and by a tight-stop
//! BOBYQA alike (they agree to 3.5e-7). Swapping the start labels makes the **base** ω the
//! one that collapses: the same defect on a plain Ω diagonal.
//!
//! Every fit here pins `optimizer = bobyqa`: `auto` may become L-BFGS for mixtures (#2749),
//! which reaches the guard on its own. BOBYQA has no gradient, so every fixture runs on
//! finite differences only; no analytic sensitivity is exercised.
//!
//! Fits to convergence, so gated (AGENTS.md, Tier 3).

use std::path::Path;

use ferx_core::estimation::parameterization::{coordinate_names, pack_params, unpack_params};
use ferx_core::io::datareader::read_nonmem_csv;
use ferx_core::types::{
    EstimationMethod, FitOptions, FitResult, FitStatus, HeldReason, Optimizer, ParameterKind,
};

const MIX_DATA: &str = "tests/data/guard_hold/mix_override_collapse.csv";

/// The internal guard of an Ω diagonal and of a `[mixture]` Ω override, in packed ln-SD.
const OMEGA_GUARD: f64 = -6.0;

/// `swap = false`: class 1 starts at the low-CL group, so the class-2 override collapses.
/// `swap = true`: class 1 starts at the high-CL (no-IIV) group, so the base ω collapses
/// and the override carries the IIV.
fn mixture_source(swap: bool) -> String {
    let (c1, c2) = if swap { (2.5, 1.2) } else { (1.2, 2.5) };
    let (base, ovr) = if swap { (0.05, 0.09) } else { (0.09, 0.05) };
    format!(
        "[parameters]
  theta TVCL1({c1}, 0.01, 100.0)
  theta TVCL2({c2}, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ {base}
  sigma EPS ~ 0.1 (sd)

[mixture]
  nsub = 2
  logit(1) = MIXL
  omega(2) ETA_CL ~ {ovr}

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
    )
}

/// The fit, its packed coordinate names and packed estimate.
fn fit_mixture(swap: bool) -> (FitResult, Vec<String>, Vec<f64>) {
    let model = ferx_core::parse_model_string(&mixture_source(swap)).expect("model parses");
    let pop = read_nonmem_csv(Path::new(MIX_DATA), None, None).expect("data loads");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        optimizer: Optimizer::Bobyqa,
        run_covariance_step: true,
        ..FitOptions::default()
    };
    let r = ferx_core::fit(&model, &pop, &model.default_params, &opts).expect("fit runs");
    let names = coordinate_names(&model.default_params);
    let packed = r.packed_estimate.clone().expect("packed estimate");
    // Round-trip through the model's own packing, as the optimizer reads it.
    let packed = pack_params(&unpack_params(&packed, &model.default_params));
    eprintln!(
        "swap={swap} ofv={:.9} converged={} stop={:?} n_iter={} status={:?}",
        r.ofv,
        r.converged,
        r.optimizer_stop,
        r.n_iterations,
        r.status()
    );
    for (n, v) in names.iter().zip(&packed) {
        eprintln!("  packed {n} = {v:.9}");
    }
    eprintln!("  held_at_bound {:?}", r.held_at_bound);
    for w in &r.warnings {
        eprintln!("  WARN {w}");
    }
    (r, names, packed)
}

fn packed_of(names: &[String], packed: &[f64], name: &str) -> f64 {
    let i = names.iter().position(|n| n == name).expect(name);
    packed[i]
}

fn rail_warning(r: &FitResult) -> &str {
    let w: Vec<&String> = r
        .warnings
        .iter()
        .filter(|w| w.contains("resumed from just above the guard"))
        .collect();
    assert_eq!(w.len(), 1, "one rail-completion warning: {:?}", r.warnings);
    w[0]
}

/// T4. The class-2 override is resumed from just above the guard and ends on it, at an OFV within
/// the measured bound of the reference optimum; the warning names it with its class.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_mixture_override_stopped_short_of_the_guard_is_resumed_onto_it() {
    let (r, names, packed) = fit_mixture(false);
    assert!(r.ofv.is_finite(), "{}", r.ofv);
    let ovr = packed_of(&names, &packed, "ETA_CL_MIX2");
    assert!(
        (ovr - OMEGA_GUARD).abs() <= 1e-12,
        "override packed {ovr}, not on the guard"
    );
    // OFV bound: see the measurement note on `OFV_BOUND_OVERRIDE`.
    assert!(r.ofv <= OFV_BOUND_OVERRIDE, "OFV {:.9}", r.ofv);
    assert!(r.converged);
    assert!(r.status() == FitStatus::Converged, "{:?}", r.status());
    let w = rail_warning(&r);
    assert!(w.contains("the random effect ETA_CL_MIX2 ("), "{w}");
    assert!(
        !w.contains("ETA_CL ("),
        "the bare base name points at the base: {w}"
    );
    // `held_at_bound` listing the override (`class = Some(2)`) needs #2831, open when this
    // was written: whichever of the two lands second adds that assertion here.
}

/// T5. Labels swapped: the base ω collapses instead, is resumed onto the guard and held;
/// the warning names `ETA_CL`, never the override.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_base_omega_stopped_short_of_the_guard_is_resumed_onto_it_and_held() {
    let (r, names, packed) = fit_mixture(true);
    assert!(r.ofv.is_finite(), "{}", r.ofv);
    let base = packed_of(&names, &packed, "ETA_CL");
    assert!(
        (base - OMEGA_GUARD).abs() <= 1e-12,
        "base ω packed {base}, not on the guard"
    );
    assert!(r.ofv <= OFV_BOUND_BASE, "OFV {:.9}", r.ofv);
    assert!(r.converged);
    assert!(
        r.held_at_bound
            .iter()
            .any(|h| h.kind == ParameterKind::Omega
                && h.index == 0
                && h.reason == HeldReason::CollapsedAtGuard),
        "{:?}",
        r.held_at_bound
    );
    assert!(r.status() == FitStatus::Converged, "{:?}", r.status());
    let w = rail_warning(&r);
    assert!(w.contains("the random effect ETA_CL ("), "{w}");
    assert!(!w.contains("ETA_CL_MIX2"), "{w}");
}

// OFV bounds, from Linux aarch64 at this PR's head (tools/linux-test.sh, glibc 2.41):
//
//   T4 override  stop −37.238760 → resumed −37.396981632  (reference −37.397489713)
//   T5 base ω    stop −37.346059 → resumed −37.396686098
//
// Each bound sits 4.8e-4 / 4.9e-4 above its measured value (BOBYQA's own `xtol` slack
// on the other coordinates is 5.1e-4 / 8.0e-4 of Newton decrement, so a platform moving
// that slack fails first here, not silently), and 0.158 / 0.050 below the stop the defect
// reports — so a fit that is not resumed cannot pass.
const OFV_BOUND_OVERRIDE: f64 = -37.3965;
const OFV_BOUND_BASE: f64 = -37.3962;
