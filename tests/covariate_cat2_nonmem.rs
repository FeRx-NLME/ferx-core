//! NONMEM 7.5.1 anchor for the `categorical2` covariate form (#1312).
//!
//! `categorical2` has an exact NONMEM spelling — the reference level's factor is
//! `1` and each non-reference level's is its own `THETA`, which is three lines of
//! `$PK` — so no exception to the NONMEM-comparison rule applies here and the
//! anchor is a plain converged run of that model.
//!
//! ## Data
//!
//! `tests/nonmem/covariate_cat2.csv` — 60 subjects, single 100 mg oral dose, ten
//! sampling times, `SEX` alternating 0/1 by subject index. Simulated **outside
//! both engines** (a Bateman closed form in Python, seed 20250909) from
//! `TVCL = 1`, a `cat2` factor of `1.5` on `CL` at `SEX = 1`, `TVV = 20`,
//! `TVKA = 1`, `ω²(CL) = ω²(V) = 0.09`, proportional `σ² = 0.04`; the committed
//! CSV is the record, so the generator never runs again.
//!
//! The effect is deliberately large and both levels are populated 30/30: a
//! factor of 1.5 on `CL` is far outside the ~0.14 SE the fit reports for it, so
//! neither engine can pass by fitting noise, and the contrast branch — the only
//! branch the two shapes differ in — is exercised by half the dataset.
//!
//! ## Two runs, because one would not see the shape
//!
//! `covariate_cat2.ctl` is the `THETA(4)` shape; `covariate_cat.ctl` is the
//! `1 + THETA(4)` shape on the same data, started at the corresponding value.
//! The pair is what makes the reparameterization identity an *external* result
//! rather than an internal one: NONMEM reaches an identical objective from both,
//! with its two θ exactly 1 apart. A single run would anchor the number but not
//! the claim that the two forms are the same model.
//!
//! ## Reference values (`tests/nonmem/covariate_cat2.{ctl,lst,ext}`)
//!
//! Converged FOCEI (`METHOD=1 INTER`) fits from the same starting values, both
//! engines, both shapes:
//!
//! | Quantity        | NONMEM `cat2` | ferx `cat2` | NONMEM `cat` | ferx `cat` |
//! |-----------------|---------------|-------------|--------------|------------|
//! | OFV             | −73.8266171   | −73.8266174 | −73.8266171  | −73.8266174 |
//! | TVCL            | 0.983305      | 0.9833024   | 0.983305     | 0.9833053  |
//! | TVV             | 21.4119       | 21.411936   | 21.4119      | 21.411948  |
//! | TVKA            | 1.03442       | 1.0344236   | 1.03442      | 1.0344269  |
//! | θ(SEX)          | 1.68330       | 1.6833125   | 0.683302     | 0.6833155  |
//! | SE θ(SEX)       | 0.137302      | 0.1371289   | 0.137302     | 0.1371289  |
//! | ω²(CL)          | 0.0827743     | 0.082772    | 0.0827743    | 0.082772   |
//! | ω²(V)           | 0.105923      | 0.105925    | 0.105923     | 0.105925   |
//! | σ² (prop)       | 0.0381895     | 0.0381894   | 0.0381895    | 0.0381894  |
//!
//! ferx reports σ as an SD; the table squares it for comparison with NONMEM's
//! `$SIGMA`, which is a variance.
//!
//! ## Tolerances
//!
//! Measured, not argued. The realised worst errors on this fit are quoted at
//! each assertion; the bounds are those numbers with an order of magnitude of
//! headroom, so a regression that moves an estimate by a percent fails rather
//! than passing inside a loose band.

use std::path::Path;

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::types::{FitOptions, FitResult};
use ferx_core::{fit, read_nonmem_csv};

const DATA: &str = "tests/nonmem/covariate_cat2.csv";

// NONMEM 7.5.1 FOCEI reference (tests/nonmem/covariate_cat2.ext, the
// `-1000000000` row; SE from the `-1000000001` row). The `cat` twin
// (tests/nonmem/covariate_cat.ext) reports every one of these identically
// except θ(SEX), which is NM_THETA_SEX_CAT2 − 1.
const NM_OFV: f64 = -73.826617109808865;
const NM_TVCL: f64 = 0.983305;
const NM_TVV: f64 = 21.4119;
const NM_TVKA: f64 = 1.03442;
const NM_THETA_SEX_CAT2: f64 = 1.68330;
const NM_THETA_SEX_CAT: f64 = 0.683302;
const NM_SE_THETA_SEX: f64 = 0.137302;
const NM_OMEGA_CL: f64 = 0.0827743;
const NM_OMEGA_V: f64 = 0.105923;
const NM_SIGMA_VAR: f64 = 0.0381895;

fn anchor_fit(model_file: &str) -> FitResult {
    let model = parse_model_file(Path::new(model_file)).expect("anchor model must parse");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("anchor data must load");
    let opts = FitOptions::default();
    fit(&model, &pop, &model.default_params, &opts).expect("anchor fit")
}

/// The named θ of a fit — the block generates it, so it is found by name rather
/// than by a position the block is free to choose.
fn theta(res: &FitResult, name: &str) -> f64 {
    let i = res
        .theta_names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("no θ named {name} in {:?}", res.theta_names));
    res.theta[i]
}

fn se_theta(res: &FitResult, name: &str) -> f64 {
    let i = res
        .theta_names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("no θ named {name} in {:?}", res.theta_names));
    res.se_theta.as_ref().expect("covariance step ran")[i]
}

/// `categorical2` reproduces NONMEM's hand-written `IF (SEX.EQ.1) COVCL =
/// THETA(4)` model, to the precision NONMEM prints.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn categorical2_matches_the_nonmem_hand_written_twin() {
    let res = anchor_fit("tests/nonmem/covariate_cat2.ferx");
    assert!(res.converged, "the anchor fit must converge");
    assert!(
        res.ofv.is_finite(),
        "a diverged or repelled fit would satisfy a bound but not the anchor: OFV {}",
        res.ofv
    );

    // Realised |ΔOFV| = 2.563e-7 against NONMEM's printed 15 digits.
    assert!(
        (res.ofv - NM_OFV).abs() < 1e-5,
        "ferx OFV {} vs NONMEM {NM_OFV}",
        res.ofv
    );

    // Realised worst relative error over the structural θ: 3.477e-6 (TVKA);
    // TVCL 2.636e-6, TVV 1.659e-6.
    // NONMEM prints six significant figures, so 1e-4 is the floor a tighter
    // bound could mean anything at.
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
    assert!(rel(theta(&res, "TVCL"), NM_TVCL) < 1e-4, "{:?}", res.theta);
    assert!(rel(theta(&res, "TVV"), NM_TVV) < 1e-4, "{:?}", res.theta);
    assert!(rel(theta(&res, "TVKA"), NM_TVKA) < 1e-4, "{:?}", res.theta);

    // The covariate θ itself: realised relative error 7.425e-6.
    let sex = theta(&res, "THETA_CL_SEX_1");
    assert!(
        rel(sex, NM_THETA_SEX_CAT2) < 1e-4,
        "θ(SEX) {sex} vs NONMEM {NM_THETA_SEX_CAT2}"
    );
    // Non-degeneracy: the effect is real and large. A factor collapsed to the
    // null (1) would still sit inside a *relative* band around a small θ; it
    // does not sit inside this one, which is ~5 SE away from 1.
    assert!(
        (sex - 1.0).abs() > 4.0 * NM_SE_THETA_SEX,
        "θ(SEX) {sex} is indistinguishable from the null factor of 1 — the fixture \
         has stopped testing the covariate"
    );

    // ferx's SE is pure R⁻¹, the same estimator as NONMEM's `MATRIX=R`.
    // Realised relative error 1.261e-3 — three orders looser than the point
    // estimates, because an SE is a second derivative of the objective and the
    // two engines build it by different quadrature.
    let se = se_theta(&res, "THETA_CL_SEX_1");
    assert!(
        rel(se, NM_SE_THETA_SEX) < 1e-2,
        "SE θ(SEX) {se} vs NONMEM {NM_SE_THETA_SEX}"
    );

    // Random effects. Realised worst relative error 3.009e-5 (ω²(CL)); ω²(V)
    // 1.617e-5, σ² 1.109e-6. ferx reports σ as an SD, NONMEM `$SIGMA` as a
    // variance.
    assert!(rel(res.omega[(0, 0)], NM_OMEGA_CL) < 1e-3, "{}", res.omega);
    assert!(rel(res.omega[(1, 1)], NM_OMEGA_V) < 1e-3, "{}", res.omega);
    let sigma_var = res.sigma[0] * res.sigma[0];
    assert!(
        rel(sigma_var, NM_SIGMA_VAR) < 1e-3,
        "σ² {sigma_var} vs NONMEM {NM_SIGMA_VAR}"
    );
}

/// The reparameterization identity, anchored *outside* ferx.
///
/// NONMEM reaches the same objective from the `cat` and the `cat2` spellings and
/// lands with its two θ exactly 1 apart (`covariate_cat.ext` vs
/// `covariate_cat2.ext`). ferx must do the same thing on the same data — and the
/// two engines must agree on both halves, not only on one.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn the_two_shapes_are_one_model_in_both_engines() {
    // The claim being anchored is a property of NONMEM's own two runs first.
    assert!(
        (NM_THETA_SEX_CAT2 - NM_THETA_SEX_CAT - 1.0).abs() < 1e-5,
        "NONMEM's own two θ must differ by 1, or there is nothing to anchor"
    );

    let cat2 = anchor_fit("tests/nonmem/covariate_cat2.ferx");
    let cat = anchor_fit("tests/nonmem/covariate_cat.ferx");
    assert!(cat.converged && cat2.converged);
    assert!(cat.ofv.is_finite() && cat2.ofv.is_finite());

    // Same objective, both engines. Realised |ΔOFV| between the two ferx fits:
    // 3.462e-9 — the two optimizer paths differ, the minimum does not.
    assert!(
        (cat.ofv - cat2.ofv).abs() < 1e-4,
        "cat {} vs cat2 {}",
        cat.ofv,
        cat2.ofv
    );
    assert!(
        (cat.ofv - NM_OFV).abs() < 1e-5,
        "cat vs NONMEM: {}",
        cat.ofv
    );

    // …and the θ exactly 1 apart, as NONMEM's pair is. Realised |Δ − 1| = 2.958e-6.
    let t_cat = theta(&cat, "THETA_CL_SEX_1");
    let t_cat2 = theta(&cat2, "THETA_CL_SEX_1");
    assert!(
        (t_cat2 - t_cat - 1.0).abs() < 1e-4,
        "cat θ {t_cat}, cat2 θ {t_cat2}: the two shapes must differ by exactly 1"
    );
    // The straddle: the two θ are genuinely different numbers, so this is not
    // two copies of one fit.
    assert!((t_cat2 - t_cat).abs() > 0.5);

    // Each engine agrees with the other on *both* shapes — a cross-engine check
    // on only the new shape could not tell a shared convention from a right one.
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
    assert!(rel(t_cat, NM_THETA_SEX_CAT) < 1e-4, "cat θ {t_cat}");
    assert!(rel(t_cat2, NM_THETA_SEX_CAT2) < 1e-4, "cat2 θ {t_cat2}");
}
