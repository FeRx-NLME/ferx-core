//! P0 regression harness for the outer-optimizer conditioning rework (issue #864).
//!
//! Fits a curated set of standard, converging example models with **default**
//! `FitOptions` and pins their converged objective + estimates against a
//! committed baseline snapshot (`tests/regression/baselines.json`). The
//! conditioning rework (#864) is an all-models change to the outer optimizer;
//! this test is the safety net that proves it moves **no** already-converging
//! fit beyond a tight tolerance.
//!
//! Scope: the "standard-fit core" — analytic/ODE PK-PD example models that fit
//! to convergence under default features against a committed CSV. Simulation-
//! only, adaptive-dosing, and feature-gated (markov / survival / nn) models are
//! deliberately excluded: they are not on the outer-optimizer path this rework
//! touches, and each carries its own anchor test. Excluded families are logged
//! by name in `EXCLUDED` below so the omission is explicit, never silent.
//!
//! Gate: slow (runs to convergence), so skipped in the default PR job and run
//! nightly via `slow-tests.yml`.
//!
//!   cargo test --features slow-tests --test regression_baseline
//!
//! ## Regenerating the baseline
//!
//! After an *intended* change (or when adding a model to the manifest), rebuild
//! the snapshot and commit the diff:
//!
//!   FERX_REGEN_BASELINE=1 cargo test --features slow-tests --test regression_baseline -- --nocapture
//!
//! The regen run writes `tests/regression/baselines.json` and does not assert.
//! Inspect the git diff before committing — an unexpected OFV move there is
//! exactly the regression this harness exists to catch.

use ferx_core::run_model_with_data;
use std::collections::BTreeMap;
use std::path::Path;

/// One model in the regression manifest: a stable key, the `.ferx` model file,
/// and the committed dataset it fits against. Each is fit exactly as the CLI
/// runs it (`run_model_with_data`), honouring the file's own `[fit_options]`
/// (method, optimizer, iov_column, …) — so the baseline reflects how the model
/// actually converges in practice, which is what "zero regression" is measured
/// against.
struct ModelSpec {
    /// Stable snapshot key (also the baseline JSON key).
    name: &'static str,
    model: &'static str,
    data: &'static str,
}

/// The standard-fit core. Keep alphabetical by `name`. To add a model, append a
/// row, regenerate the baseline, and commit both together.
const MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "one_cpt_infusion",
        model: "examples/one_cpt_infusion.ferx",
        data: "data/one_cpt_infusion.csv",
    },
    ModelSpec {
        name: "one_cpt_iv",
        model: "examples/one_cpt_iv.ferx",
        data: "data/one_cpt_iv.csv",
    },
    ModelSpec {
        name: "two_cpt_iv",
        model: "examples/two_cpt_iv.ferx",
        data: "data/two_cpt_iv.csv",
    },
    ModelSpec {
        name: "warfarin",
        model: "examples/warfarin.ferx",
        data: "data/warfarin.csv",
    },
    ModelSpec {
        name: "warfarin_additive_eta",
        model: "examples/warfarin_additive_eta.ferx",
        data: "data/warfarin_additive_eta.csv",
    },
    ModelSpec {
        name: "warfarin_addl",
        model: "examples/warfarin_addl.ferx",
        data: "data/warfarin_addl.csv",
    },
    ModelSpec {
        name: "warfarin_block_omega",
        model: "examples/warfarin_block_omega.ferx",
        data: "data/warfarin_block_omega.csv",
    },
    ModelSpec {
        name: "warfarin_iov",
        model: "examples/warfarin_iov.ferx",
        data: "data/warfarin_iov.csv",
    },
    ModelSpec {
        name: "warfarin_ltbs",
        model: "examples/warfarin_ltbs.ferx",
        data: "data/warfarin_ltbs.csv",
    },
    ModelSpec {
        name: "warfarin_ss",
        model: "examples/warfarin_ss.ferx",
        data: "data/warfarin_ss.csv",
    },
];

/// Model families intentionally left out of the standard-fit core, with the
/// reason. Documented here so the exclusion is explicit (CLAUDE.md: "no silent
/// caps"). These are not on the outer-optimizer regression path or need special
/// features/harnesses; each has its own dedicated anchor test.
#[allow(dead_code)]
const EXCLUDED: &[(&str, &str)] = &[
    (
        "adaptive_*",
        "feedback dosing — no static fit; own anchor tests",
    ),
    ("ctmm_*", "requires --features markov"),
    (
        "rtte_* / tte_*",
        "survival endpoints — requires --features survival",
    ),
    ("warfarin_dcm", "requires --features nn"),
    (
        "*_saem / *_bayes / warfarin_impmap",
        "stochastic estimator — not the FOCEI outer path",
    ),
    ("warfarin_sde", "SDE — forced-FD gradient, separate path"),
    (
        "warfarin_scaled",
        "stale example: gradient_method = ad (retired) — does not run",
    ),
    (
        "simulation-only examples",
        "no committed dataset to fit against",
    ),
];

/// Snapshot of one converged fit. Floats are compared with tolerances below.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct Baseline {
    converged: bool,
    ofv: f64,
    /// theta name -> value.
    theta: BTreeMap<String, f64>,
    /// omega diagonal (variances), in eta order.
    omega_diag: Vec<f64>,
    /// sigma name -> value.
    sigma: BTreeMap<String, f64>,
}

/// Tolerances. OFV is the primary conditioning signal — pin it tightly. Params
/// are weakly identified on some models (fluconazole Q, #864), so a looser
/// relative band avoids false alarms while still catching a basin change.
const OFV_ABS_TOL: f64 = 1e-3;
const OFV_REL_TOL: f64 = 1e-6;
const PARAM_REL_TOL: f64 = 1e-3;
const PARAM_ABS_TOL: f64 = 1e-6;

fn baseline_path() -> &'static str {
    "tests/regression/baselines.json"
}

fn run_fit(spec: &ModelSpec) -> Result<Baseline, String> {
    let (res, _pop) = run_model_with_data(spec.model, Some(spec.data))?;
    let theta = res
        .theta_names
        .iter()
        .cloned()
        .zip(res.theta.iter().copied())
        .collect();
    let sigma = res
        .sigma_names
        .iter()
        .cloned()
        .zip(res.sigma.iter().copied())
        .collect();
    let n = res.omega.nrows();
    let omega_diag = (0..n).map(|i| res.omega[(i, i)]).collect();
    Ok(Baseline {
        converged: res.converged,
        ofv: res.ofv,
        theta,
        omega_diag,
        sigma,
    })
}

fn close(actual: f64, expected: f64, abs_tol: f64, rel_tol: f64) -> bool {
    let diff = (actual - expected).abs();
    diff <= abs_tol || diff <= rel_tol * expected.abs()
}

/// Compare a fresh fit against its baseline; return a list of human-readable
/// discrepancies (empty == pass).
fn diff_against(name: &str, got: &Baseline, base: &Baseline) -> Vec<String> {
    let mut issues = Vec::new();
    if got.converged != base.converged {
        issues.push(format!(
            "{name}: converged {} -> {} (was {}, now {})",
            base.converged, got.converged, base.converged, got.converged
        ));
    }
    if !close(got.ofv, base.ofv, OFV_ABS_TOL, OFV_REL_TOL) {
        issues.push(format!(
            "{name}: OFV {:.6} -> {:.6} (Δ {:.3e})",
            base.ofv,
            got.ofv,
            got.ofv - base.ofv
        ));
    }
    for (k, bv) in &base.theta {
        match got.theta.get(k) {
            Some(gv) if close(*gv, *bv, PARAM_ABS_TOL, PARAM_REL_TOL) => {}
            Some(gv) => issues.push(format!("{name}: theta {k} {bv:.6} -> {gv:.6}")),
            None => issues.push(format!("{name}: theta {k} missing in new fit")),
        }
    }
    if got.omega_diag.len() != base.omega_diag.len() {
        issues.push(format!(
            "{name}: omega dim {} -> {}",
            base.omega_diag.len(),
            got.omega_diag.len()
        ));
    } else {
        for (i, (g, b)) in got.omega_diag.iter().zip(&base.omega_diag).enumerate() {
            if !close(*g, *b, PARAM_ABS_TOL, PARAM_REL_TOL) {
                issues.push(format!("{name}: omega[{i}] {b:.6} -> {g:.6}"));
            }
        }
    }
    for (k, bv) in &base.sigma {
        match got.sigma.get(k) {
            Some(gv) if close(*gv, *bv, PARAM_ABS_TOL, PARAM_REL_TOL) => {}
            Some(gv) => issues.push(format!("{name}: sigma {k} {bv:.6} -> {gv:.6}")),
            None => issues.push(format!("{name}: sigma {k} missing in new fit")),
        }
    }
    issues
}

fn load_baselines() -> BTreeMap<String, Baseline> {
    let raw = std::fs::read_to_string(baseline_path()).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e}\nrun with FERX_REGEN_BASELINE=1 first",
            baseline_path()
        )
    });
    serde_json::from_str(&raw).expect("baselines.json is not valid JSON")
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn regression_baseline_holds() {
    let regen = std::env::var("FERX_REGEN_BASELINE").is_ok();

    if regen {
        let mut snapshot: BTreeMap<String, Baseline> = BTreeMap::new();
        let mut failed = Vec::new();
        for spec in MODELS {
            match run_fit(spec) {
                Ok(b) => {
                    println!(
                        "regen {}: ofv={:.4} converged={} thetas={} omega={} sigma={}",
                        spec.name,
                        b.ofv,
                        b.converged,
                        b.theta.len(),
                        b.omega_diag.len(),
                        b.sigma.len()
                    );
                    snapshot.insert(spec.name.to_string(), b);
                }
                Err(e) => failed.push(format!("{}: {e}", spec.name)),
            }
        }
        let json = serde_json::to_string_pretty(&snapshot).unwrap();
        std::fs::create_dir_all(Path::new(baseline_path()).parent().unwrap()).unwrap();
        std::fs::write(baseline_path(), json).unwrap();
        println!("wrote {} baselines to {}", snapshot.len(), baseline_path());
        if !failed.is_empty() {
            panic!(
                "these manifest models did NOT fit and were omitted from the baseline:\n  {}",
                failed.join("\n  ")
            );
        }
        return;
    }

    let baselines = load_baselines();
    let mut all_issues = Vec::new();
    for spec in MODELS {
        let Some(base) = baselines.get(spec.name) else {
            all_issues.push(format!(
                "{}: no baseline entry — regenerate with FERX_REGEN_BASELINE=1",
                spec.name
            ));
            continue;
        };
        match run_fit(spec) {
            Ok(got) => all_issues.extend(diff_against(spec.name, &got, base)),
            Err(e) => all_issues.push(format!("{}: fit errored: {e}", spec.name)),
        }
    }
    assert!(
        all_issues.is_empty(),
        "regression harness detected {} discrepancy(ies):\n  {}",
        all_issues.len(),
        all_issues.join("\n  ")
    );
}
