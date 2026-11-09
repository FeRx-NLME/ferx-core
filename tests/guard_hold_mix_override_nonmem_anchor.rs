//! #2809's NONMEM anchor: a collapsed `[mixture]` per-class override that ferx holds at
//! its internal guard, against NONMEM with that override `FIX`.
//!
//! `nonmem_anchor/guard_hold_mix_override.ctl` is the #2809 fixture
//! (`tests/data/guard_hold/mix_override_collapse.csv`) at ferx's FOCEI estimate, with the
//! class-2 CL variance as a second ETA used only for `MIXNUM.EQ.2`, `FIX` at
//! `6.1442123533282098E-6`, `$EST MAXEVAL=0`, `$COV MATRIX=R SIGL=12` (NONMEM 7.6.0, `nm3`
//! anchor image). "Held" means FIX at the estimate, which NONMEM computes.
//!
//! Unlike #2797's anchor, this one **does** separate held from unheld, on two of its six
//! SEs: ferx's unheld SEs (before #2809) differ from the held ones by 9.3e-3 relative on σ
//! and 5.37e-3 on TVCL2, against a NONMEM gap of 4.0e-4 and 4e-7. On TVCL1 (3.2e-4) the
//! unheld shift is inside the bound and NONMEM does not tell the two apart. Tier 2: one
//! evaluation, not gated.

use std::path::PathBuf;

use ferx_core::run_model_with_data;
use ferx_core::types::{FitResult, HeldReason, ParameterKind};

const DATA: &str = "tests/data/guard_hold/mix_override_collapse.csv";
const EXT: &str = "nonmem_anchor/results/guard_hold_mix_override.ext";
const SIGMA_SD: f64 = 0.10298957033853946;

const MODEL: &str = "[parameters]
  theta TVCL1(1.2921355862556707, 0.01, 100.0)
  theta TVCL2(2.526598380988927, 0.01, 100.0)
  theta TVV(10.040303488207734, 0.1, 1000.0)
  theta MIXL(-0.11953563542469542, -10.0, 10.0)
  omega ETA_CL ~ 0.06227533113210565
  sigma EPS ~ 0.10298957033853946 (sd)

[mixture]
  nsub = 2
  logit(1) = MIXL
  omega(2) ETA_CL ~ 6.1442123533282098e-6

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)

[fit_options]
  method     = focei
  maxiter    = 0
  covariance = true
";

/// The `.ext` row `iteration` (−1000000000 estimates, −1000000001 SEs), by column name.
fn ext_row(iteration: &str) -> std::collections::HashMap<String, f64> {
    let text = std::fs::read_to_string(EXT).expect("the committed .ext");
    let mut lines = text.lines().skip(1);
    let header: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    let row: Vec<&str> = lines
        .find(|l| l.split_whitespace().next() == Some(iteration))
        .expect("the row")
        .split_whitespace()
        .collect();
    header
        .iter()
        .zip(&row)
        .map(|(h, v)| (h.to_string(), v.parse::<f64>().unwrap()))
        .collect()
}

fn evaluate() -> FitResult {
    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join("guard_hold_mix_override.ferx");
    std::fs::write(&path, MODEL).unwrap();
    let (fit, _) = run_model_with_data(path.to_str().unwrap(), Some(DATA)).expect("evaluate");
    fit
}

#[test]
fn held_override_standard_errors_match_nonmem_with_the_override_fixed() {
    let f = evaluate();
    let est = ext_row("-1000000000");
    let se = ext_row("-1000000001");

    // Same point, same objective. Measured: ferx −37.391596589835 (macOS arm64), NONMEM
    // −37.391596611526303 (Δ 2.2e-8).
    let d_ofv = (f.ofv - est["OBJ"]).abs();
    assert!(
        d_ofv < 1e-6,
        "OFV {} vs NONMEM {} (Δ {d_ofv:e})",
        f.ofv,
        est["OBJ"]
    );

    // ferx held the class-2 override; NONMEM had it FIX.
    assert_eq!(f.held_at_bound.len(), 1, "{:?}", f.held_at_bound);
    let h = &f.held_at_bound[0];
    assert_eq!(
        (h.name.as_str(), h.kind, h.index, h.class, h.reason),
        (
            "ETA_CL_MIX2",
            ParameterKind::Omega,
            0,
            Some(2),
            HeldReason::CollapsedAtGuard
        )
    );
    assert!(f.cov_fixed[6], "{:?}", f.cov_fixed);
    assert_eq!(
        se["OMEGA(2,2)"], 1e10,
        "NONMEM reports no SE for a FIX element"
    );

    let st = f.se_theta.as_ref().unwrap();
    let so = f.se_omega.as_ref().unwrap();
    // σ: NONMEM's SIGMA is the variance σ²; ferx stores the SD. SE(σ) = SE(σ²) / 2σ.
    let s_sd = f.se_sigma.as_ref().unwrap()[0];
    let pairs = [
        ("THETA1", st[0], se["THETA1"]),
        ("THETA2", st[1], se["THETA2"]),
        ("THETA3", st[2], se["THETA3"]),
        ("THETA4", st[3], se["THETA4"]),
        ("OMEGA(1,1)", so[0], se["OMEGA(1,1)"]),
        ("SIGMA(1,1)", s_sd, se["SIGMA(1,1)"] / (2.0 * SIGMA_SD)),
    ];
    let mut worst = (0.0_f64, "");
    for (name, ferx, nm) in pairs {
        assert!(
            ferx.is_finite() && nm.is_finite() && nm > 0.0,
            "{name}: {ferx} vs {nm}"
        );
        let rel = (ferx - nm).abs() / nm;
        eprintln!("{name}: ferx {ferx:.6e}  NONMEM {nm:.5e}  rel {rel:.2e}");
        if rel > worst.0 {
            worst = (rel, name);
        }
    }
    // Measured (macOS arm64 and Linux aarch64, identical in every printed digit): worst
    // 3.99e-4 at SIGMA(1,1), then 1.97e-4 (THETA1), 8.1e-5
    // (OMEGA(1,1)), the rest ≤ 1.5e-5; the .ext prints 6 significant digits. NONMEM's
    // finite-difference `$COV` against ferx's analytic R. 2× headroom. The unheld step
    // (before #2809) moves SIGMA(1,1) by 9.3e-3 and THETA2 by 5.37e-3 — 11× and 6.7× the
    // bound — so a step that stops holding the override fails here.
    const BOUND: f64 = 8e-4;
    assert!(
        worst.0 < BOUND,
        "worst relative SE gap {:.3e} at {}",
        worst.0,
        worst.1
    );
}
