//! #2833's NONMEM anchor: the standard errors of a `[mixture]` fit's per-class overrides
//! — reported in `FitResult::se_mixture_overrides` and labelled by
//! `FitResult::mixture_override_params` — against NONMEM at the same point.
//!
//! Fixture P (`tests/data/mixture/mix_override_se.py`, copied unchanged as
//! `nonmem_anchor/mix_override_se.csv`): 2 classes, 80 subjects × 7 observations, both
//! overrides **interior**, unlike #2809's on the rail. ferx evaluates at its own FOCEI
//! estimate (`maxiter = 0`); `nonmem_anchor/mix_override_se.ctl` is the same point with
//! each override as a second ETA / EPS used only for `MIXNUM.EQ.2`, `$EST MAXEVAL=0`,
//! `$COV MATRIX=R SIGL=8` (NONMEM 7.6.0, `nm3` anchor image).
//!
//! **SIGL=8, not 12.** NONMEM's finite-difference R drifts on this model as its step
//! shrinks: against ferx's analytic R, SIGL=8 agrees on all eight SEs to ≤ 7.1e-5, SIGL=12
//! is 1 % off on TVCL2 and 0.66 % on `EPS_MIX2`, SIGL=14 9 % off on TVCL2. ferx's analytic
//! and finite-difference R agree with each other and with SIGL=8, so the drift is
//! NONMEM's step, not ferx (the sweep is in `nonmem_anchor/README.md`). Tier 2: one
//! evaluation each, not gated.

use std::path::{Path, PathBuf};

use ferx_core::types::{FitResult, ParameterKind};
use ferx_core::{run_covariance, run_model_with_data, FitOptions};

const DATA: &str = "nonmem_anchor/mix_override_se.csv";
const COV: &str = "nonmem_anchor/results/mix_override_se.cov";
const EXT: &str = "nonmem_anchor/results/mix_override_se.ext";
/// The two σ (SD) at the evaluated point: NONMEM's SIGMA is the variance, so
/// `SE(σ) = SE(σ²) / 2σ`.
const SIGMA_SD: [f64; 2] = [0.10051824793268943, 0.21117117356992268];
/// Per-SE relative bounds, in [`nonmem_ses`] order (see the table on each test).
const BOUNDS_ANALYTIC: [f64; 8] = [2e-5, 1.5e-4, 2e-5, 2e-5, 2e-5, 5e-5, 2e-5, 9e-5];
const BOUNDS_FD: [f64; 8] = [4e-4, 3e-5, 8e-3, 1.2e-3, 5e-4, 1.2e-3, 6e-4, 9e-4];

/// Fixture P at ferx's FOCEI estimate. `sigma(2) … (sd)`: without `(sd)` the class σ is
/// read on the variance scale, like an undecorated base σ (OFV 597.1, not 441.58).
fn model(analytic_cov_hessian: bool, sigma2_fix: bool) -> String {
    format!(
        "[parameters]
  theta TVCL1(1.2047967374230335, 0.01, 100.0)
  theta TVCL2(2.6436593127545596, 0.01, 100.0)
  theta TVV(10.00618251814227, 0.1, 1000.0)
  theta MIXL(-0.032037516843225214, -10.0, 10.0)
  omega ETA_CL ~ 0.0607653415782817
  sigma EPS ~ 0.10051824793268943 (sd)

[mixture]
  nsub = 2
  logit(1) = MIXL
  omega(2) ETA_CL ~ 0.266883174876294
  sigma(2) EPS ~ 0.21117117356992268 (sd){fix}

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)

[fit_options]
  method     = focei
  covariance = true
  maxiter    = 0
  analytic_cov_hessian = {analytic_cov_hessian}
",
        fix = if sigma2_fix { " FIX" } else { "" }
    )
}

/// Evaluate fixture P; the model file lives in `dir`, which the caller keeps alive for a
/// later `run_covariance` that re-reads it.
fn evaluate(dir: &Path, analytic_cov_hessian: bool, sigma2_fix: bool) -> FitResult {
    let path: PathBuf = dir.join("mix_override_se.ferx");
    std::fs::write(&path, model(analytic_cov_hessian, sigma2_fix)).unwrap();
    let (fit, _) = run_model_with_data(path.to_str().unwrap(), Some(DATA)).expect("evaluate");
    fit
}

/// A whitespace table of NONMEM's (`.ext` / `.cov`): header names → the row whose first
/// cell is `first`.
fn nm_row(path: &str, first: &str) -> std::collections::HashMap<String, f64> {
    let text = std::fs::read_to_string(path).expect("the committed NONMEM output");
    let mut lines = text.lines().skip(1);
    let header: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    let row: Vec<&str> = lines
        .find(|l| l.split_whitespace().next() == Some(first))
        .unwrap_or_else(|| panic!("no `{first}` row in {path}"))
        .split_whitespace()
        .collect();
    header
        .iter()
        .zip(&row)
        .skip(1)
        .map(|(h, v)| (h.to_string(), v.parse::<f64>().unwrap()))
        .collect()
}

/// NONMEM's SE of each `.cov` coordinate, from the diagonal, on ferx's scales.
fn nonmem_ses() -> Vec<(&'static str, f64)> {
    let diag = |name: &str| nm_row(COV, name)[name].sqrt();
    vec![
        ("TVCL1", diag("THETA1")),
        ("TVCL2", diag("THETA2")),
        ("TVV", diag("THETA3")),
        ("MIXL", diag("THETA4")),
        ("ETA_CL", diag("OMEGA(1,1)")),
        ("EPS", diag("SIGMA(1,1)") / (2.0 * SIGMA_SD[0])),
        ("ETA_CL_MIX2", diag("OMEGA(2,2)")),
        ("EPS_MIX2", diag("SIGMA(2,2)") / (2.0 * SIGMA_SD[1])),
    ]
}

/// ferx's SEs in [`nonmem_ses`] order.
fn ferx_ses(f: &FitResult) -> Vec<f64> {
    let st = f.se_theta.as_ref().expect("se_theta");
    let ov = f
        .se_mixture_overrides
        .as_ref()
        .expect("se_mixture_overrides");
    vec![
        st[0],
        st[1],
        st[2],
        st[3],
        f.se_omega.as_ref().unwrap()[0],
        f.se_sigma.as_ref().unwrap()[0],
        ov[0],
        ov[1],
    ]
}

/// Each ferx SE against NONMEM's, each against its own relative bound.
fn assert_against_nonmem(f: &FitResult, bounds: [f64; 8], engine: &str) {
    let mut failed = Vec::new();
    for (((name, nm), ferx), bound) in nonmem_ses().into_iter().zip(ferx_ses(f)).zip(bounds) {
        assert!(
            ferx.is_finite() && nm.is_finite() && nm > 0.0,
            "{engine} {name}: {ferx} vs {nm}"
        );
        let rel = (ferx - nm).abs() / nm;
        eprintln!(
            "{engine} {name}: ferx {ferx:.9e}  NONMEM {nm:.6e}  rel {rel:.2e}  bound {bound:.1e}"
        );
        if rel >= bound {
            failed.push(format!("{name}: rel {rel:.3e} ≥ {bound:.1e}"));
        }
    }
    assert!(failed.is_empty(), "{engine}: {}", failed.join("; "));
}

/// The descriptors, the objective, and that the fixture never left the analytic path.
fn assert_fixture(f: &FitResult) {
    // Same point, same objective: ferx 441.58074542, NONMEM 441.58074550704185 (Δ 9e-8).
    let ofv = nm_row(EXT, "-1000000000")["OBJ"];
    assert!((f.ofv - ofv).abs() < 1e-6, "OFV {} vs NONMEM {ofv}", f.ofv);
    let got: Vec<_> = f
        .mixture_override_params
        .iter()
        .map(|p| {
            (
                p.name.as_str(),
                p.kind,
                p.index,
                p.class,
                p.packed_index,
                p.fixed,
            )
        })
        .collect();
    assert_eq!(
        got,
        [
            ("ETA_CL_MIX2", ParameterKind::Omega, 0, 2, 6, false),
            ("EPS_MIX2", ParameterKind::Sigma, 0, 2, 7, false),
        ]
    );
    assert!(f.held_at_bound.is_empty(), "{:?}", f.held_at_bound);
    assert!(
        !f.warnings
            .iter()
            .any(|w| w.contains("finite-difference inner gradients")),
        "the fixture fell back to FD inner gradients: {:?}",
        f.warnings
    );
}

/// T7 (#2833): the analytic R's override SEs — and the six base SEs beside them —
/// against NONMEM SIGL=8, and the labels every renderer prints on the real fit.
///
/// Bounds: 2× each SE's measured gap, floored at 2e-5 (the `.cov` prints 6 significant
/// digits). Measured on Linux aarch64 (the reference platform, `tools/linux-test.sh`) and
/// macOS arm64, identical in every printed digit:
///
///   SE            measured   bound
///   TVCL1         6.15e-6    2e-5
///   TVCL2         7.16e-5    1.5e-4
///   TVV           4.15e-6    2e-5
///   MIXL          9.76e-6    2e-5
///   ETA_CL        1.12e-6    2e-5
///   EPS           2.19e-5    5e-5
///   ETA_CL_MIX2   3.80e-6    2e-5
///   EPS_MIX2      4.10e-5    9e-5
///
/// Mutations: read `se_mixture_overrides` from a neighbouring row (κ forgotten, Ω/Σ
/// swapped) or drop the Ω factor 2 → `ETA_CL_MIX2` / `EPS_MIX2` red by orders of
/// magnitude; move any base SE → its own row red.
#[test]
fn override_standard_errors_match_nonmem_analytic_r() {
    let dir = tempfile::tempdir().unwrap();
    let f = evaluate(dir.path(), true, false);
    assert_fixture(&f);
    assert_against_nonmem(&f, BOUNDS_ANALYTIC, "analytic");

    // The labels on the real fit: no `param_` padding in the YAML, the override row in
    // the table.
    let yaml_path = dir.path().join("fit.yaml");
    ferx_core::io::output::write_estimates_yaml(&f, yaml_path.to_str().unwrap()).unwrap();
    let yaml = std::fs::read_to_string(&yaml_path).unwrap();
    assert!(
        yaml.contains(
            "  parameters: [TVCL1, TVCL2, TVV, MIXL, log_chol_ETA_CL, EPS, \
             log_chol_ETA_CL_MIX2, EPS_MIX2]\n"
        ),
        "{yaml}"
    );
    assert!(!yaml.contains("param_"), "{yaml}");
    let table = ferx_core::io::output::parameter_table(&f);
    let row = table
        .lines()
        .find(|l| l.starts_with("ETA_CL_MIX2 "))
        .unwrap_or_else(|| panic!("{table}"));
    assert!(
        row.contains(" 0.266883 ") && row.contains(" 0.064976 ") && row.ends_with("(class 2)"),
        "{row}"
    );
}

/// T8 (#2833): the same anchor on the finite-difference R stencil
/// (`analytic_cov_hessian = false`), each SE against its own bound — the stencil's own,
/// not T7's: the FD R sits up to 3.75e-3 (TVV) from NONMEM, so T7's bounds would fail it.
/// The override gaps (2.7e-4, 4.4e-4) are the stencil's accuracy, not a scale error: a
/// dropped factor 2 or a neighbouring row is off by ≥ 50 %. Measured on Linux aarch64 and
/// macOS arm64, identical in every printed digit; bounds 2× the gap:
///
///   SE            measured   bound
///   TVCL1         1.93e-4    4e-4
///   TVCL2         1.26e-5    3e-5
///   TVV           3.75e-3    8e-3
///   MIXL          5.58e-4    1.2e-3
///   ETA_CL        2.19e-4    5e-4
///   EPS           5.73e-4    1.2e-3
///   ETA_CL_MIX2   2.74e-4    6e-4
///   EPS_MIX2      4.44e-4    9e-4
#[test]
fn override_standard_errors_match_nonmem_fd_r() {
    let dir = tempfile::tempdir().unwrap();
    let f = evaluate(dir.path(), false, false);
    assert_fixture(&f);
    assert_against_nonmem(&f, BOUNDS_FD, "fd");
}

/// T9 (#2833): a standalone `run_covariance` on the fit refreshes `se_mixture_overrides`
/// and reproduces the inline step's bit for bit. Mutation: skip the `run_covariance.rs`
/// site — the refreshed fit carries the input's vector, which this clears first.
#[test]
fn run_covariance_refreshes_override_standard_errors() {
    let dir = tempfile::tempdir().unwrap();
    let f = evaluate(dir.path(), true, false);
    let inline = f.se_mixture_overrides.clone().expect("inline SEs");
    let mut stale = f.clone();
    stale.se_mixture_overrides = None;
    let out = run_covariance(&stale, None, None, &FitOptions::default()).expect("run_covariance");
    let again = out.se_mixture_overrides.expect("run_covariance SEs");
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert!(
        inline.iter().all(|s| s.is_finite() && *s > 0.0),
        "{inline:?}"
    );
    assert_eq!(bits(&again), bits(&inline), "{again:?} vs {inline:?}");
    assert_eq!(out.se_theta, f.se_theta);
    assert_eq!(out.mixture_override_params, f.mixture_override_params);
}

/// T11 (#2833): a user `FIX` on the σ override makes it `fixed` — from the model's FIX
/// mask — and it renders `---` / `[FIX]`, never a number, with no `held:` key (it is not
/// in `held_at_bound`). With T10 (`tests/guard_hold_mix_override_nonmem_anchor.rs`) both
/// sides of the mask choice are pinned: here the covariance step's `cov_fixed` agrees, so
/// reading `fixed` from it would pass; there it would fail. Mutation: `fixed` from
/// nothing (always `false`) → the row prints its SE and this test dies.
#[test]
fn user_fixed_override_renders_fixed() {
    let dir = tempfile::tempdir().unwrap();
    let f = evaluate(dir.path(), true, true);
    let p = &f.mixture_override_params[1];
    assert_eq!((p.name.as_str(), p.packed_index), ("EPS_MIX2", 7));
    assert!(
        p.fixed && f.cov_fixed[7],
        "fixed {} cov_fixed {}",
        p.fixed,
        f.cov_fixed[7]
    );
    assert!(!f.mixture_override_params[0].fixed);
    assert!(f.held_at_bound.is_empty(), "{:?}", f.held_at_bound);
    let se = f.se_mixture_overrides.as_ref().unwrap();
    assert_eq!(se[1], 0.0);
    assert!(se[0].is_finite() && se[0] > 0.0, "{se:?}");

    let summary = ferx_core::io::output::format_summary(&f);
    let row = summary
        .lines()
        .find(|l| l.trim_start().starts_with("EPS_MIX2 [FIX]"))
        .unwrap_or_else(|| panic!("{summary}"));
    assert!(row.ends_with("SE = ---"), "{row}");
    let table = ferx_core::io::output::parameter_table(&f);
    let row = table
        .lines()
        .find(|l| l.starts_with("EPS_MIX2 [FIX]"))
        .unwrap_or_else(|| panic!("{table}"));
    assert!(row.contains(" --- ") && !row.contains("0.000000"), "{row}");
    let path = dir.path().join("fit.yaml");
    ferx_core::io::output::write_estimates_yaml(&f, path.to_str().unwrap()).unwrap();
    let yaml = std::fs::read_to_string(&path).unwrap();
    assert!(
        yaml.contains("    kind: sigma\n    estimate: 0.211171\n    fixed: true\n    se: ~\n"),
        "{yaml}"
    );
    assert!(!yaml.contains("held:"), "{yaml}");
}
