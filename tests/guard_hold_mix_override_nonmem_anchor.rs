//! #2809's NONMEM anchor: a collapsed `[mixture]` per-class override that ferx holds at
//! its internal guard, against NONMEM with that override `FIX`.
//!
//! `nonmem_anchor/guard_hold_mix_override.ctl` is the #2809 fixture
//! (`tests/data/guard_hold/mix_override_collapse.csv`) at ferx's FOCEI estimate, with the
//! class-2 CL variance as a second ETA used only for `MIXNUM.EQ.2`, `FIX` at
//! `6.1442123533282098E-6`, `$EST MAXEVAL=0`, `$COV MATRIX=R SIGL=12` (NONMEM 7.6.0, `nm3`
//! anchor image). "Held" means FIX at the estimate, which NONMEM computes.
//!
//! Unlike #2797's anchor, this one **does** separate held from unheld, on five of its six
//! SEs, each against its own bound (the table in the test): ferx's unheld SEs (before
//! #2809) move TVCL2 by 5.37e-3 and σ by 9.3e-3, against NONMEM gaps of 4e-7 and 4.0e-4.
//! On TVCL1 (3.2e-4) the unheld shift is inside the bound and NONMEM does not tell the two
//! apart. Tier 2: one evaluation, not gated.

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
    // One bound per SE, from its own measured gap (#2831 review r1 row 1), so the tight
    // ones are not loosened by σ's. Measured on macOS arm64 and Linux aarch64, identical
    // in every printed digit; NONMEM's finite-difference `$COV` against ferx's analytic R.
    // Each bound is 2× its measured gap, floored at 2e-5: the `.ext` prints 6 significant
    // digits, a rounding of at most 5e-6 relative, so a bound below 2e-5 would test
    // NONMEM's print format. The unheld step (before #2809) moves THETA2 by 5.37e-3 (268×
    // its bound), SIGMA(1,1) by 9.3e-3 (11.6×), OMEGA(1,1) by 7.4e-4 (4.6×), THETA3 by
    // 7.0e-4 (23×) and THETA4 by 7.2e-5 (3.6×); THETA1's 3.2e-4 shift is inside its 4e-4.
    //
    //   SE          measured   bound
    //   THETA1      1.97e-4    4e-4
    //   THETA2      4.1e-7     2e-5
    //   THETA3      1.51e-5    3e-5
    //   THETA4      4.5e-7     2e-5
    //   OMEGA(1,1)  8.05e-5    1.6e-4
    //   SIGMA(1,1)  3.99e-4    8e-4
    let pairs = [
        ("THETA1", st[0], se["THETA1"], 4e-4),
        ("THETA2", st[1], se["THETA2"], 2e-5),
        ("THETA3", st[2], se["THETA3"], 3e-5),
        ("THETA4", st[3], se["THETA4"], 2e-5),
        ("OMEGA(1,1)", so[0], se["OMEGA(1,1)"], 1.6e-4),
        (
            "SIGMA(1,1)",
            s_sd,
            se["SIGMA(1,1)"] / (2.0 * SIGMA_SD),
            8e-4,
        ),
    ];
    let mut failed = Vec::new();
    for (name, ferx, nm, bound) in pairs {
        assert!(
            ferx.is_finite() && nm.is_finite() && nm > 0.0,
            "{name}: {ferx} vs {nm}"
        );
        let rel = (ferx - nm).abs() / nm;
        eprintln!("{name}: ferx {ferx:.6e}  NONMEM {nm:.5e}  rel {rel:.2e}  bound {bound:.1e}");
        if rel >= bound {
            failed.push(format!("{name}: rel {rel:.3e} ≥ {bound:.1e}"));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("; "));
}

/// T10 (#2833): the held override renders as held — `N/A` in the console and the table,
/// `se: ~` + `held: collapsed_at_guard` in the YAML — while its base `ETA_CL` keeps its
/// number. NONMEM prints no SE for the `FIX` `OMEGA(2,2)` (asserted above).
///
/// The straddle that makes `fixed`'s source observable: the covariance step held this
/// coordinate (`cov_fixed[6]`), yet it is **not** fixed by the model (`fixed == false`).
/// Mutations: read `fixed` from `cov_fixed` → the row renders `---` (fixed), not `N/A`;
/// drop the held arm of `override_rows` → the row prints its SE (0.0). How an entry is
/// matched to an override, `(class, kind, index)`, is pinned by `held_matching_is_by_class`
/// (`src/io/output_mixture_override_tests.rs`), not here: this fixture has one override and
/// no base entry, so a looser match renders the same rows.
#[test]
fn held_override_renders_held_and_its_base_keeps_its_se() {
    let f = evaluate();
    let p = &f.mixture_override_params[0];
    assert_eq!((p.name.as_str(), p.packed_index), ("ETA_CL_MIX2", 6));
    assert!(
        f.cov_fixed[p.packed_index] && !p.fixed,
        "straddle lost: cov_fixed {} fixed {}",
        f.cov_fixed[p.packed_index],
        p.fixed
    );

    let summary = ferx_core::io::output::format_summary(&f);
    let line = |text: &str, name: &str| -> String {
        text.lines()
            .find(|l| l.trim_start().split_whitespace().next() == Some(name))
            .unwrap_or_else(|| panic!("no `{name}` row in:\n{text}"))
            .to_string()
    };
    assert!(
        line(&summary, "ETA_CL_MIX2").ends_with("SE = N/A"),
        "{summary}"
    );
    let base = line(&summary, "ETA_CL");
    assert!(!base.contains("N/A") && !base.contains("---"), "{summary}");

    let table = ferx_core::io::output::parameter_table(&f);
    assert!(line(&table, "ETA_CL_MIX2").contains(" N/A "), "{table}");
    assert!(!line(&table, "ETA_CL").contains("N/A"), "{table}");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fit.yaml");
    ferx_core::io::output::write_estimates_yaml(&f, path.to_str().unwrap()).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    // The top-level sections; the `estimates:` block after them repeats them (#2911).
    let yaml = &text[..text.find("\nestimates:\n").expect("estimates block")];
    let section = &yaml[yaml.find("\nmixture_overrides:\n").expect("section")..];
    assert!(
        section.starts_with(
            "\nmixture_overrides:\n  ETA_CL_MIX2:\n    class: 2\n    base: ETA_CL\n    \
             kind: omega\n"
        ),
        "{section}"
    );
    assert!(
        section.contains("    se: ~\n    held: collapsed_at_guard\n"),
        "{section}"
    );
    assert_eq!(
        yaml.matches("held: collapsed_at_guard").count(),
        1,
        "{yaml}"
    );
}
