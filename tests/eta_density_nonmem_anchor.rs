//! NONMEM reduction anchors for the omega-line random-effect densities (#1716, #1721
//! review finding 4).
//!
//! NONMEM has no non-Gaussian random-effect prior, but the same marginal model can be
//! written in it with a standard-normal η (`$OMEGA 1 FIX`) read through a map:
//!
//! | stream | ferx | NONMEM `$PK` | comparison |
//! |---|---|---|---|
//! | `eta_density_lognormal` | `omega ETA_CL ~ 0.2 density lognormal`, `CL = TVCL * ETA_CL` | `CL = TVCL*EXP(SQRT(S)*ETA(1) − S/2)`, `S = ln 1.2` | FOCEI, `MAXEVAL=0`: an **affine** map of η, so the Laplace / FOCEI objective is the same function — to arithmetic |
//! | `eta_density_logistic` | `omega ETA_CL ~ 0.2 density logistic`, `CL = TVCL * exp(ETA_CL)` | `X = B*LOG(P/(1−P))`, `P = PHI(ETA(1))` | IMP, `EONLY=1`: not affine, so only the exact marginal agrees — within Monte Carlo error |
//!
//! The NONMEM outputs are committed in `nonmem_anchor/results/` (run in
//! `nonmem:7.6.0-anchor` with `nmfe76` directly, since `nm3` cannot read a `MAXEVAL=0`
//! or `EONLY` objective):
//!
//! - **lognormal**: `the_lognormal_density_matches_nonmem` checks ferx's density OFV
//!   against `.ext`'s `OBJ` and every subject's EBE against `.phi`, ferx's `z` mapped
//!   to NONMEM's η by `(z + S/2)/√S`. Measured: OFV 6.4e-8, EBE 2.4e-8.
//! - **logistic**: two IMP engines differ by ≈ 0.06 on a *Gaussian* model already
//!   (`eta_shape_null` under the same `$EST`), so the absolute −2LL is the wrong object.
//!   The density's contribution is the difference against that null, which agreed to
//!   0.007 in the review run (NONMEM's SE of the difference ≈ 0.025).
//!   `the_logistic_density_matches_nonmem_net_of_the_null` asserts it once the null's
//!   IMP output (`eta_density_imp_null.ctl`) is committed.
//!
//! `the_lognormal_density_is_its_affine_gaussian_twin` runs the first reduction inside
//! ferx as well: the density path against ferx's own Gaussian path.

use std::collections::HashMap;
use std::path::Path;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{FitOptions, FitResult};
use ferx_core::{fit, read_nonmem_csv};

const DATA: &str = "nonmem_anchor/eta_shape.csv";

/// The anchor model of `eta_shape_null.ctl` with `omega_cl` and `CL` replaced.
fn model(omega_cl: &str, cl: &str, extra_theta: &str) -> String {
    format!(
        "
[parameters]
  theta TVCL(2.0, 0.01, 100.0)
  theta TVV(20.0, 0.1, 1000.0)
{extra_theta}
  {omega_cl}
  omega ETA_V ~ 0.1
  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = {cl}
  V = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"
    )
}

/// The FOCEI evaluation at the declared parameters (`outer_maxiter = 0` is `MAXEVAL=0`).
fn evaluate(src: &str) -> FitResult {
    let parsed = parse_full_model(src).expect("the ferx model must parse");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("the dataset must load");
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let r = fit(&parsed.model, &pop, &parsed.model.default_params, &options)
        .expect("the evaluation must run");
    assert!(r.ofv.is_finite(), "OFV is not finite");
    r
}

/// `density lognormal` (mean 1, variance ω = 0.2) against the Gaussian stream NONMEM runs,
/// `TVCL·exp(√S·η − S/2)`, η ~ N(0, 1) FIX, `S = ln(1 + ω)`: ferx's latent is
/// `z = √S·η − S/2`, an affine map, under which the FOCEI objective is invariant (the
/// prior's `−2 ln |a|` and the curvature's `+2 ln |a|` cancel). So the OFVs agree to the
/// inner optimiser's tolerance and every subject's EBE maps onto the other's.
/// Dies under: a wrong latent mean (`−S/2`) or variance (`S = ln(1 + ω)`, not ω) in the
/// MvLogNormal / log-normal prior, or its curvature left out of `H̃`.
#[test]
fn the_lognormal_density_is_its_affine_gaussian_twin() {
    let s = 0.2f64.ln_1p();
    let density = evaluate(&model(
        "omega ETA_CL ~ 0.2 density lognormal",
        "TVCL * ETA_CL",
        "",
    ));
    let twin = evaluate(&model(
        "omega ETA_CL ~ 1.0 FIX",
        "TVCL * exp(sqrt(S) * ETA_CL - S / 2)",
        &format!("  theta S({s:?}, 0.0001, 10.0, FIX)"),
    ));
    let gap = (density.ofv - twin.ofv).abs();
    eprintln!(
        "lognormal density OFV {} vs affine Gaussian twin {}: |Δ| {gap:.2e}",
        density.ofv, twin.ofv
    );
    let mut worst_eta = 0.0f64;
    for (d, t) in density.subjects.iter().zip(&twin.subjects) {
        let (z, eta) = (d.eta[0], t.eta[0]);
        let mapped = (z + s / 2.0) / s.sqrt();
        assert!(
            mapped.is_finite() && eta.is_finite(),
            "subject {}: {mapped} vs {eta}",
            d.id
        );
        worst_eta = worst_eta.max((mapped - eta).abs());
    }
    eprintln!("worst |(z + S/2)/√S − η| {worst_eta:.2e}");
    assert!(gap < OFV_TOL, "|Δ OFV| {gap:e}");
    assert!(worst_eta < ETA_TOL, "worst EBE gap {worst_eta:e}");
}
// Measured: |Δ OFV| 5.08e-8 and worst EBE gap 3.94e-8 (the inner optimiser's
// tolerance, the two parameterisations converging separately); bounds 20× / 25×.
const OFV_TOL: f64 = 1e-6;
const ETA_TOL: f64 = 1e-6;

/// Rows of a committed NONMEM output, as `header → column` maps, `skip` lines in.
fn nonmem_rows(file: &str, skip: usize) -> Vec<HashMap<String, f64>> {
    let text = std::fs::read_to_string(Path::new("nonmem_anchor/results").join(file))
        .unwrap_or_else(|e| panic!("{file} must be readable: {e}"));
    let mut lines = text.lines().skip(skip);
    let header: Vec<String> = lines
        .next()
        .expect("a header line")
        .split_whitespace()
        .map(str::to_string)
        .collect();
    lines
        .filter(|l| !l.trim_start().starts_with("TABLE"))
        .map(|l| {
            header
                .iter()
                .cloned()
                .zip(l.split_whitespace().map(|v| v.parse().expect("a number")))
                .collect()
        })
        .collect()
}

/// The `OBJ` of a `.ext` row whose `ITERATION` is `iter`.
fn ext_obj(file: &str, iter: f64) -> f64 {
    nonmem_rows(file, 1)
        .into_iter()
        .find(|r| r["ITERATION"] == iter)
        .unwrap_or_else(|| panic!("{file}: no ITERATION {iter}"))["OBJ"]
}

/// `density lognormal` against NONMEM 7.6.0's affine Gaussian stream
/// (`eta_density_lognormal.ctl`, FOCEI `MAXEVAL=0`): the OFV and every subject's EBE on
/// both ETAs, ferx's latent mapped to NONMEM's η by `(z + S/2)/√S`.
/// Dies under: the same defects as the twin test, now against an external engine.
#[test]
fn the_lognormal_density_matches_nonmem() {
    let s = 0.2f64.ln_1p();
    let r = evaluate(&model(
        "omega ETA_CL ~ 0.2 density lognormal",
        "TVCL * ETA_CL",
        "",
    ));
    let nm = ext_obj("eta_density_lognormal.ext", -1_000_000_000.0);
    assert!(nm.is_finite(), "NONMEM OBJ {nm}");
    let gap = (r.ofv - nm).abs();
    let phi = nonmem_rows("eta_density_lognormal.phi", 1);
    let mut worst_eta = 0.0f64;
    for subj in &r.subjects {
        let id: f64 = subj.id.parse().expect("numeric IDs");
        let row = phi.iter().find(|p| p["ID"] == id).expect("a .phi row");
        let got = [(subj.eta[0] + s / 2.0) / s.sqrt(), subj.eta[1]];
        for (k, g) in got.into_iter().enumerate() {
            let want = row[&format!("ETA({})", k + 1)];
            assert!(
                g.is_finite() && want.is_finite(),
                "subject {id} η{k}: {g} vs {want}"
            );
            worst_eta = worst_eta.max((g - want).abs());
        }
    }
    eprintln!(
        "lognormal vs NONMEM: |Δ OFV| {gap:.2e} (ferx {}, NONMEM {nm}), worst EBE {worst_eta:.2e}",
        r.ofv
    );
    // Measured: 6.4e-8 and 2.4e-8 (Linux aarch64, the review run); bounds ~15×.
    assert!(gap < 1e-6, "|Δ OFV| {gap:e}");
    assert!(worst_eta < 4e-7, "worst EBE {worst_eta:e}");
}

/// `density logistic` against NONMEM's `F⁻¹(Φ(η))` stream under IMP, **net of a Gaussian
/// null under the same IMP** (`eta_density_imp_null.ctl`): two IMP engines differ by ≈ 0.06
/// on the null already, so only the density's contribution is comparable. NONMEM's side is
/// the mean over its `EONLY` iterations, ferx's the mean over five seeds.
#[test]
#[ignore = "pending: commit nonmem_anchor/results/eta_density_imp_null.ext (eta_density_imp_null.ctl in nonmem:7.6.0-anchor)"]
fn the_logistic_density_matches_nonmem_net_of_the_null() {
    let mean_iters = |file: &str| {
        let objs: Vec<f64> = nonmem_rows(file, 1)
            .into_iter()
            .filter(|r| r["ITERATION"] >= 0.0)
            .map(|r| r["OBJ"])
            .collect();
        assert!(
            objs.iter().all(|v| v.is_finite()) && !objs.is_empty(),
            "{file}"
        );
        objs.iter().sum::<f64>() / objs.len() as f64
    };
    let nm_diff = mean_iters("eta_density_logistic.ext") - mean_iters("eta_density_imp_null.ext");
    let imp = |omega_cl: &str| {
        let parsed = parse_full_model(&model(omega_cl, "TVCL * exp(ETA_CL)", "")).expect("parses");
        let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("the dataset must load");
        let ofvs: Vec<f64> = (1..=5u64)
            .map(|seed| {
                let options = FitOptions {
                    method: ferx_core::types::EstimationMethod::Imp,
                    imp_eval_only: true,
                    imp_samples: 20_000,
                    imp_seed: Some(seed),
                    run_covariance_step: false,
                    ..parsed.fit_options.clone()
                };
                fit(&parsed.model, &pop, &parsed.model.default_params, &options)
                    .expect("IMP evaluates")
                    .ofv
            })
            .collect();
        assert!(ofvs.iter().all(|v| v.is_finite()), "{ofvs:?}");
        ofvs.iter().sum::<f64>() / ofvs.len() as f64
    };
    let ferx_diff = imp("omega ETA_CL ~ 0.2 density logistic") - imp("omega ETA_CL ~ 0.2");
    let gap = (ferx_diff - nm_diff).abs();
    eprintln!("logistic − null: ferx {ferx_diff:.4}, NONMEM {nm_diff:.4}, gap {gap:.4}");
    // Review run: gap 0.007 against NONMEM's SE of the difference ≈ 0.025; 3 SE.
    assert!(gap < 0.075, "gap {gap}");
}
