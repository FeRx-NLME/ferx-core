//! #2797's NONMEM anchor: a coordinate ferx holds at its internal guard against NONMEM
//! with that coordinate `FIX`.
//!
//! `nonmem_anchor/guard_hold.ctl` is fixture A (`tests/data/guard_hold/`) at ferx's FOCEI
//! estimate with `$OMEGA 6.1442123533282098E-6 FIX` on KA, `$EST MAXEVAL=0`,
//! `$COV MATRIX=R SIGL=12` (NONMEM 7.6.0, `nm3` anchor image). "Held" means FIX at the
//! estimate, and that is what NONMEM computes, so its standard errors are the external
//! reference for ferx's conditional ones. The `N/A` for the held coordinate itself is a
//! display choice with no NONMEM analogue.
//!
//! What this anchor cannot see: at this stationary point the conditional and the
//! unconditional SEs differ by at most 5.4e-6 relative (the plan's §0), far inside the
//! ferx↔NONMEM gap below, so it does not tell *held* from *not held*. That differential is
//! `guard_hold.rs`'s, against the same engine's FIX clone (bit-equal) and the poor-σ
//! straddle. Tier 2: one evaluation, not gated.

use std::path::PathBuf;

use ferx_core::run_model_with_data;
use ferx_core::types::{FitResult, HeldReason};

const DATA: &str = "tests/data/guard_hold/sim_oral_no_ka_iiv.csv";
const EXT: &str = "nonmem_anchor/results/guard_hold.ext";
const SIGMA_SD: f64 = 0.10069194732313741;

const MODEL: &str = "[parameters]
  theta TVCL(0.12250099819205257, 0.001, 10.0)
  theta TVV(7.753670975867449, 0.1, 500.0)
  theta TVKA(0.9982755334099482, 0.01, 50.0)
  omega ETA_CL ~ 0.0792688562315328
  omega ETA_V  ~ 0.05944155243978235
  omega ETA_KA ~ 6.1442123533282098e-6
  sigma PROP_ERR ~ 0.10069194732313741 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

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
    let path: PathBuf = dir.path().join("guard_hold.ferx");
    std::fs::write(&path, MODEL).unwrap();
    let (fit, _) = run_model_with_data(path.to_str().unwrap(), Some(DATA)).expect("evaluate");
    fit
}

#[test]
fn held_standard_errors_match_nonmem_with_the_coordinate_fixed() {
    let f = evaluate();
    let est = ext_row("-1000000000");
    let se = ext_row("-1000000001");

    // Same point, same objective. Measured: ferx 353.754540470156, NONMEM
    // 353.75454052250410 (Δ 5.2e-8).
    let d_ofv = (f.ofv - est["OBJ"]).abs();
    assert!(
        d_ofv < 1e-6,
        "OFV {} vs NONMEM {} (Δ {d_ofv:e})",
        f.ofv,
        est["OBJ"]
    );

    // ferx held ETA_KA; NONMEM had it FIX.
    assert_eq!(f.held_at_bound.len(), 1);
    assert_eq!(f.held_at_bound[0].name, "ETA_KA");
    assert_eq!(f.held_at_bound[0].reason, HeldReason::CollapsedAtGuard);
    assert!(f.cov_fixed[5], "{:?}", f.cov_fixed);
    assert_eq!(
        se["OMEGA(3,3)"], 1e10,
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
        ("OMEGA(1,1)", so[0], se["OMEGA(1,1)"]),
        ("OMEGA(2,2)", so[1], se["OMEGA(2,2)"]),
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
    // Measured (macOS arm64): worst 1.25e-3 at THETA2, every other SE ≤ 5.9e-4. The .ext
    // prints 6 significant digits (≤ 1e-5 relative), so the gap is the two R matrices —
    // NONMEM's finite-difference `$COV` against ferx's exact analytic R — not print
    // precision. 2× headroom. A step that stopped conditioning moves these by ≤ 5.4e-6
    // and cannot fail this bound; one that dropped a coordinate or conditioned on the
    // wrong one moves them by orders more.
    const BOUND: f64 = 2.5e-3;
    assert!(
        worst.0 < BOUND,
        "worst relative SE gap {:.3e} at {}",
        worst.0,
        worst.1
    );
}
