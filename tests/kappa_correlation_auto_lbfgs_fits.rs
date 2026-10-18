//! Tier 3 for #2122: on a model with a kappa and a residual `correlation` line, both
//! gradients are finite differences (#2101), and `optimizer = auto` used to resolve to
//! BOBYQA, whose `FtolReached` stopped the fit at a different optimum for each starting
//! value of ρ — two of three above the optimum of the nested kappa-only model, which the
//! correlated model contains at ρ = 0. `auto` now takes L-BFGS on that route.
//!
//! Each fixture is fitted from several starting values of ρ under the default `auto`.
//! The fits must agree with each other, and none may end above the nested kappa-only fit.
//! The numbers in the bounds are Linux (`tools/linux-test.sh`), see each constant.
//!
//! - `tests/data/mbma_placebo_corr/` — ferx-r's `mbma_placebo` example (synthetic data,
//!   generator in ferx-r's `inst/examples/mbma_placebo/simulate_dataset.R`), plus
//!   `theta RHO` and `correlation car1(RHO) by OCC`: the reporter's model.
//! - `nonmem_anchor/nested_levels_3.ferx` on `nested_levels_3_ar1_ferx.csv`, plus a free
//!   `RHO` and `correlation ar1(RHO) by ARM`: the #2102 anchor's model, fitted.

use ferx_core::api::{bind_theta_levels, read_population_for};
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, EstimationMethod, FitOptions, FitResult};

const MBMA_MODEL: &str = include_str!("data/mbma_placebo_corr/mbma_placebo.ferx");
const MBMA_DATA: &str = "tests/data/mbma_placebo_corr/mbma_placebo.csv";
const EXACT3_MODEL: &str = include_str!("../nonmem_anchor/nested_levels_3.ferx");
const EXACT3_DATA: &str = "nonmem_anchor/nested_levels_3_ar1/nested_levels_3_ar1_ferx.csv";

fn run(src: &str, data: &str) -> FitResult {
    let mut parsed = parse_full_model(src).expect("model parses");
    let (mut pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        data,
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("data loads");
    // mbma_placebo's `theta PLACEBO[STUDY, TIME]` takes its levels from the data.
    bind_theta_levels(&mut parsed, src, &mut pop).expect("levels bind");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    fit(&parsed.model, &pop, &parsed.model.default_params, &opts).expect("fit runs")
}

/// Insert `line` after `anchor`, which must occur exactly once. A bare `replacen` is a
/// silent no-op when the anchor text moves (`nested_levels_3.ferx` is a shared anchor
/// file): without the correlation line the fixture is the kappa-only model, both bounds
/// pass, and the test no longer tests the route (#2127 review finding 4).
fn insert_after(src: &str, anchor: &str, line: &str) -> String {
    assert_eq!(
        src.matches(anchor).count(),
        1,
        "anchor {anchor:?} must occur once"
    );
    src.replacen(anchor, &format!("{anchor}{line}"), 1)
}

fn mbma_car1(rho0: f64) -> String {
    let src = insert_after(
        MBMA_MODEL,
        "[parameters]\n",
        &format!("  theta RHO({rho0}, 0.0, 0.95)\n"),
    );
    insert_after(
        &src,
        "  DV ~ additive(ADD_ERR) weight = SE\n",
        "  correlation car1(RHO) by OCC\n",
    )
}

fn exact3_ar1(rho0: f64) -> String {
    let src = insert_after(
        EXACT3_MODEL,
        "  sigma ADD ~ 0.255164\n",
        &format!("  theta RHO({rho0}, -0.99, 0.99)\n"),
    );
    insert_after(
        &src,
        "  DV ~ additive(ADD)\n",
        "  correlation ar1(RHO) by ARM\n",
    )
}

fn rho_hat(r: &FitResult) -> f64 {
    let i = r.theta_names.iter().position(|n| n == "RHO").expect("RHO");
    r.theta[i]
}

/// The shared assertions: every OFV is finite, the starts agree within `spread_tol`, none
/// ends above the nested fit by more than `nested_tol`, and every start ran L-BFGS. The
/// optimizer label is asserted **last**, so that `auto` going back to BOBYQA (the
/// regression this file exists to catch) is caught by the numbers first; see each
/// fixture's constants for what BOBYQA measured.
fn check(
    label: &str,
    fits: &[(f64, FitResult)],
    nested: &FitResult,
    spread_tol: f64,
    nested_tol: f64,
) {
    assert!(nested.ofv.is_finite(), "{label}: nested OFV {}", nested.ofv);
    for (rho0, r) in fits {
        eprintln!(
            "{label}: start ρ = {rho0}: OFV = {:.10}, ρ̂ = {:.6}, optimizer = {}, converged = {}",
            r.ofv,
            rho_hat(r),
            r.optimizer,
            r.converged
        );
        // Before the fold: `f64::min`/`max` discard NaN.
        assert!(r.ofv.is_finite(), "{label}: start {rho0}: OFV {}", r.ofv);
    }
    eprintln!("{label}: nested kappa-only OFV = {:.10}", nested.ofv);
    let lo = fits
        .iter()
        .map(|(_, r)| r.ofv)
        .fold(f64::INFINITY, f64::min);
    let hi = fits
        .iter()
        .map(|(_, r)| r.ofv)
        .fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "{label}: spread = {:.3e}, worst − nested = {:.3e}",
        hi - lo,
        hi - nested.ofv
    );
    assert!(
        hi - lo <= spread_tol,
        "{label}: starts disagree by {}",
        hi - lo
    );
    assert!(
        hi <= nested.ofv + nested_tol,
        "{label}: worst start {hi} is above the nested optimum {}",
        nested.ofv
    );
    for (rho0, r) in fits {
        assert_eq!(
            r.optimizer, "auto (nlopt_lbfgs)",
            "{label}: start {rho0}: auto must take L-BFGS on the kappa + correlation route"
        );
    }
}

/// Linux aarch64 (`tools/linux-test.sh`) and x86_64 (`slow-tests.yml` run 39152008553), the
/// same in every printed digit, `auto` → L-BFGS: OFV 89.4512783684 / 89.4512784177 /
/// 89.4512784903 from ρ₀ = 0.01 / 0.3 / 0.6 (ρ̂ = 0.7598), spread 1.219e-7. `1e-5` is 82×
/// above it. Under BOBYQA (explicit, Linux aarch64) the three stop at 104.7159 / 111.6049 /
/// 98.3146: spread 13.29, 1.3e6× this bound.
const MBMA_SPREAD_TOL: f64 = 1e-5;
/// The nested kappa-only fit is 97.0700361594 on Linux; the L-BFGS worst is 7.619 below it.
/// No slack: the car1 model contains it at ρ = 0. All three BOBYQA stops end above it, the
/// nearest (98.3146) by 1.24.
const MBMA_NESTED_TOL: f64 = 0.0;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn mbma_placebo_car1_reaches_one_optimum_from_every_start() {
    let fits: Vec<(f64, FitResult)> = [0.01, 0.3, 0.6]
        .into_iter()
        .map(|rho0| (rho0, run(&mbma_car1(rho0), MBMA_DATA)))
        .collect();
    let nested = run(MBMA_MODEL, MBMA_DATA);
    check(
        "mbma_placebo + car1",
        &fits,
        &nested,
        MBMA_SPREAD_TOL,
        MBMA_NESTED_TOL,
    );
}

/// Linux aarch64 and x86_64 (`slow-tests.yml` run 39152008553), the same in every printed
/// digit, `auto` → L-BFGS: OFV 105.6397483118 / 105.6397790273 from ρ₀ = −0.3 / 0.5
/// (ρ̂ = 0.1152), spread 3.072e-5. `1e-3` is 33× above it. Under BOBYQA (Linux aarch64) the
/// two stop at 105.7361 / 105.6736: spread 0.0625, 62× this bound. That spread is the only check here
/// a BOBYQA revert reaches: both BOBYQA stops are still below the nested optimum.
const EXACT3_SPREAD_TOL: f64 = 1e-3;
/// The nested kappa-only fit is 107.0435709196 on Linux (NONMEM's optimum, the start of
/// `nested_levels_3.ferx`); the L-BFGS worst is 1.404 below it.
const EXACT3_NESTED_TOL: f64 = 0.0;

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn nested_exact3_ar1_reaches_one_optimum_from_every_start() {
    let fits: Vec<(f64, FitResult)> = [-0.3, 0.5]
        .into_iter()
        .map(|rho0| (rho0, run(&exact3_ar1(rho0), EXACT3_DATA)))
        .collect();
    let nested = run(EXACT3_MODEL, EXACT3_DATA);
    check(
        "exact3 + ar1",
        &fits,
        &nested,
        EXACT3_SPREAD_TOL,
        EXACT3_NESTED_TOL,
    );
}
