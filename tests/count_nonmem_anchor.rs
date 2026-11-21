//! The `[count_model]` Poisson / negative-binomial endpoint (#2877) against NONMEM and R,
//! through the public `fit()`.
//!
//! Fixtures: `nonmem_anchor/count_pois.csv` / `count_nb.csv` (60 subjects × 6 records,
//! `nonmem_anchor/simulate_count.R`) and `count_pois.ferx` / `count_nb.ferx`:
//!
//! ```text
//! log λ = log(BASE) + BETA·(WT/70 − 1) + GAM·TRT + ETA_C        (NB: size r = R)
//! ```
//!
//! WT varies between subjects and TRT within each subject (0 then 1), so the mean reads a
//! covariate that changes across records as well as across subjects (non-degeneracy).
//!
//! NONMEM codes each record as `Y = −2·log PMF` with `GAMLN` (the `ln Γ(k+1)` constant kept)
//! under `$EST METHOD=COND LAPLACE -2LL`, which is the objective ferx's Laplace / FOCEI /
//! FOCE compute on a count row. The fixed-effects-only arms (`count_*_fixed.ctl`, `$OMEGA 0
//! FIX`) are the same likelihood as R's `glm(family = poisson)` / `MASS::glm.nb`, and agree
//! with them to 3e-7 (`count_r_reference.out`).
//!
//! **Tier 2** (every PR): ferx evaluates once (`outer_maxiter = 0`) at NONMEM's optimum and
//! at the `lme4::glmer` optimum. **Tier 3** (`slow-tests`): ferx fits to convergence from the
//! model file's initial estimates and must land on NONMEM's optimum.
//!
//! One external reference disagrees, and is recorded rather than used: at its own optimum
//! `lme4::glmer.nb` reports −2 log L = 1697.2560, while NONMEM evaluates its Laplace
//! objective at that same point to 1697.4436 and ferx to the same digits as NONMEM. glmer.nb
//! fits `r` in an outer loop that ended with "iteration limit reached", and its `logLik` is not
//! this Laplace objective to 0.19; the Poisson `glmer` (no outer loop) agrees with NONMEM to
//! 2.5e-4.

#![cfg(feature = "survival")]

use ferx_core::types::{EstimationMethod, FitOptions, FitResult, ModelParameters, OmegaMatrix};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The final row of `nonmem_anchor/results/<name>.ext`: `n_theta` θ, `OMEGA(1,1)`, OBJ.
fn nonmem_row(name: &str, n_theta: usize) -> (Vec<f64>, f64, f64) {
    let text =
        std::fs::read_to_string(root().join(format!("nonmem_anchor/results/{name}.ext"))).unwrap();
    let mut names: Vec<String> = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.first() {
            Some(&"ITERATION") => names = fields.iter().map(|s| s.to_string()).collect(),
            Some(&"-1000000000") => {
                let get = |name: &str| -> f64 {
                    let i = names.iter().position(|n| n == name).unwrap();
                    fields[i].parse().unwrap()
                };
                let theta = (1..=n_theta).map(|k| get(&format!("THETA{k}"))).collect();
                return (theta, get("OMEGA(1,1)"), get("OBJ"));
            }
            _ => {}
        }
    }
    panic!("{name}.ext: no final row");
}

/// `count_<fam>.ferx` with `edit` applied, loaded against its data.
fn prep(fam: &str, edit: impl Fn(String) -> String) -> ferx_core::PreparedRun {
    let text =
        std::fs::read_to_string(root().join(format!("nonmem_anchor/count_{fam}.ferx"))).unwrap();
    let text = edit(text);
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("ferx_count_{fam}_{}_{n}.ferx", std::process::id()));
    std::fs::write(&path, text).unwrap();
    let p = ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(root().join(format!("nonmem_anchor/count_{fam}.csv")).to_str().unwrap()),
    )
    .unwrap_or_else(|e| panic!("count_{fam} loads: {e}"));
    let _ = std::fs::remove_file(&path);
    p
}

/// The fixed-effects-only model: the random effect removed from the file.
fn no_eta(t: String) -> String {
    assert!(t.contains("  omega ETA_C ~ 0.1\n") && t.contains(" + ETA_C"), "fixture moved");
    t.replace("  omega ETA_C ~ 0.1\n", "").replace(" + ETA_C", "")
}

fn options(p: &ferx_core::PreparedRun, method: EstimationMethod, maxiter: usize) -> FitOptions {
    FitOptions {
        verbose: false,
        outer_maxiter: maxiter,
        run_covariance_step: false,
        inner_tol: 1e-10,
        method,
        interaction: method != EstimationMethod::Foce,
        ..p.parsed.fit_options.clone()
    }
}

fn evaluate(
    p: &ferx_core::PreparedRun, theta: &[f64], omega: Option<f64>, method: EstimationMethod,
) -> FitResult {
    let mut params: ModelParameters = p.init_params.clone();
    params.theta = theta.to_vec();
    if let Some(w) = omega {
        params.omega = OmegaMatrix::from_diagonal(&[w], vec!["ETA_C".into()]);
    }
    ferx_core::fit(&p.parsed.model, &p.population, &params, &options(p, method, 0))
        .unwrap_or_else(|e| panic!("{method:?}: {e}"))
}

/// No warning may say a gradient or Hessian left the exact path: every fixture here must run
/// on the count term's compiled plan, not on finite differences.
fn assert_no_fd(what: &str, f: &FitResult) {
    for w in &f.warnings {
        assert!(
            !w.contains("finite difference") && !w.contains("finite-difference"),
            "{what}: {w}"
        );
    }
}

/// |ΔOFV| bound against NONMEM at NONMEM's own (θ, ω). Measured on macOS arm64, identical
/// under Laplace / FOCEI / FOCE: Poisson random-intercept +2.1e-8, NB random-intercept
/// −6.8e-9, Poisson fixed −1.2e-10, NB fixed +4.5e-11, Poisson at the glmer point −3.6e-9,
/// NB at the glmer.nb point −3.3e-9. The fixed arms are exact likelihoods; the random-effect
/// arms carry NONMEM's finite-difference η-Hessian in its Laplace term. The bound keeps ~10×
/// headroom over the worst; scoring the NB optimum as a Poisson moves the objective by 179
/// (`the_anchor_tells_negbin_from_poisson`), and dropping `ln Γ(k+1)` would move the Poisson
/// one by `Σ 2·ln k!` = 3134.
const NONMEM_OFV: f64 = 2e-7;

/// A-count: ferx evaluated at each NONMEM optimum reproduces NONMEM's objective, for both
/// families, with and without the random intercept, under every Laplace-form method — on the
/// exact (plan) derivatives.
fn assert_reproduces_nonmem(method: EstimationMethod) {
    for (fam, n_theta) in [("pois", 3), ("nb", 4)] {
        for kind in ["laplace", "fixed", "glmer"] {
            let name = format!("count_{fam}_{kind}");
            let (theta, omega, obj) = nonmem_row(&name, n_theta);
            let f = if kind == "fixed" {
                assert_eq!(omega, 0.0);
                evaluate(&prep(fam, no_eta), &theta, None, method)
            } else {
                evaluate(&prep(fam, |t| t), &theta, Some(omega), method)
            };
            let d = f.ofv - obj;
            eprintln!("A-count {name} {method:?}: ferx {:.10}, NONMEM {obj:.10}, Δ {d:.3e}", f.ofv);
            assert!(f.ofv.is_finite(), "{name}: {}", f.ofv);
            assert!(d.abs() <= NONMEM_OFV, "{name} {method:?}: Δ {d:.3e}");
            assert_no_fd(&name, &f);
        }
    }
}

#[test]
fn count_endpoint_reproduces_nonmem_laplace() {
    assert_reproduces_nonmem(EstimationMethod::Laplace);
}

#[test]
fn count_endpoint_reproduces_nonmem_under_focei() {
    assert_reproduces_nonmem(EstimationMethod::FoceI);
}

#[test]
fn count_endpoint_reproduces_nonmem_under_foce() {
    assert_reproduces_nonmem(EstimationMethod::Foce);
}

/// −2 log L from `nonmem_anchor/count_r_reference.out`: R's `glm` / `MASS::glm.nb` at their
/// optimum (fixed effects), and `lme4::glmer(family = poisson, nAGQ = 1)` at its optimum. In
/// R's parameterisation: intercept = log BASE, `W` = BETA, `TRT` = GAM, glm.nb's θ = r.
struct RRef {
    fam: &'static str,
    fixed: bool,
    beta: [f64; 3],
    r: Option<f64>,
    omega: Option<f64>,
    m2ll: f64,
    tol: f64,
}

/// The fixed-effects arms are an exact likelihood (no integral) — measured 4.3e-11 (glm) and
/// 1.4e-11 (glm.nb), bound 1e-9. The glmer arm is a Laplace objective whose deviance lme4
/// reports after its own PIRLS solve: measured −2.50e-4 against ferx, and NONMEM sits on
/// ferx's side of it (ferx − NONMEM = −3.6e-9 at that point), so the gap is lme4's; bound
/// 5e-4.
const R_REFS: [RRef; 3] = [
    RRef {
        fam: "pois",
        fixed: true,
        beta: [1.529062877944759, 0.534055300855542, -0.418354050896050],
        r: None,
        omega: None,
        m2ll: 1775.3799165101,
        tol: 1e-9,
    },
    RRef {
        fam: "nb",
        fixed: true,
        beta: [1.569613294969761, 0.412037719918334, -0.620067415699343],
        r: Some(1.53854432369662),
        omega: None,
        m2ll: 1739.1620527867,
        tol: 1e-9,
    },
    RRef {
        fam: "pois",
        fixed: false,
        beta: [1.376415130873234, 0.678942099471635, -0.418354051110100],
        r: None,
        omega: Some(0.269389262185741),
        m2ll: 1514.5138914680,
        tol: 5e-4,
    },
];

/// Cross-check against R: the fixed-effects case matches `glm` / `glm.nb` to printed
/// precision, and the random-intercept Poisson case matches `lme4::glmer` (Laplace).
#[test]
fn count_endpoint_matches_r_glm_and_glmer() {
    for r in &R_REFS {
        let p = if r.fixed { prep(r.fam, no_eta) } else { prep(r.fam, |t| t) };
        let mut theta = vec![r.beta[0].exp(), r.beta[1], r.beta[2]];
        theta.extend(r.r);
        let f = evaluate(&p, &theta, r.omega, EstimationMethod::Laplace);
        let d = f.ofv - r.m2ll;
        eprintln!("R {} fixed={}: ferx {:.10}, R {:.10}, Δ {d:.3e}", r.fam, r.fixed, f.ofv, r.m2ll);
        assert!(d.abs() <= r.tol, "{} fixed={}: Δ {d:.3e}", r.fam, r.fixed);
    }
}

/// The anchor sees the family: the NB optimum scored as a Poisson (one word changed in the
/// same file, at the same point) is far from NONMEM's objective, so an NB arm that scored the
/// Poisson law could not pass the tests above.
#[test]
fn the_anchor_tells_negbin_from_poisson() {
    let (theta, omega, obj) = nonmem_row("count_nb_laplace", 4);
    let to_poisson = |t: String| {
        assert!(t.contains("  type       = negbin\n") && t.contains("  r          = R\n"));
        t.replace("  type       = negbin\n", "  type       = poisson\n")
            .replace("  r          = R\n", "")
    };
    let p = prep("nb", to_poisson);
    let f = evaluate(&p, &theta, Some(omega), EstimationMethod::Laplace);
    eprintln!("NB optimum scored as Poisson: ΔOFV {:.3}", f.ofv - obj);
    assert!((f.ofv - obj).abs() > 10.0, "{} vs {obj}", f.ofv);
}

/// Tier 3: the full fit from the model file's start lands on NONMEM's optimum. Measured on
/// macOS arm64 (Laplace, the file's method): Poisson OFV 1.3e-7 below NONMEM, NB 4.3e-8
/// below. NONMEM's minimizations end in ERROR=133/134 (rounding) at a point ferx matches, so
/// the OFV bound is one-sided (ferx may not be worse by more than 1e-4) and the parameter
/// bound is 2e-3 relative, the precision a stop that flat pins θ and ω to.
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
fn count_endpoint_converges_to_nonmem_optimum() {
    for (fam, n_theta) in [("pois", 3), ("nb", 4)] {
        let p = prep(fam, |t| t);
        let (theta, omega, obj) = nonmem_row(&format!("count_{fam}_laplace"), n_theta);
        let mut opts = options(&p, EstimationMethod::Laplace, 300);
        opts.outer_maxiter = 300;
        let f = ferx_core::fit(&p.parsed.model, &p.population, &p.init_params, &opts).unwrap();
        let d = f.ofv - obj;
        eprintln!("count_{fam} converged: ferx {:.8}, NONMEM {obj:.8}, Δ {d:.3e}", f.ofv);
        assert!(d <= 1e-4, "count_{fam}: ferx worse than NONMEM by {d:.3e}");
        for (k, (a, b)) in f.theta.iter().zip(&theta).enumerate() {
            assert!((a - b).abs() <= 2e-3 * b.abs(), "count_{fam} θ{k}: {a} vs {b}");
        }
        let w = f.omega[(0, 0)];
        assert!((w - omega).abs() <= 2e-3 * omega, "count_{fam} ω: {w} vs {omega}");
        assert_no_fd(fam, &f);
    }
}
