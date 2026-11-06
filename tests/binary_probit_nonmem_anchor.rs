//! The `[binary_model]` probit link and `pnorm` (#2603), through the public `fit()`. Every
//! fit is an evaluation (`outer_maxiter = 0`), so the file is Tier 2.
//!
//! **A-B1-probit.** `nonmem_anchor/probit_b1_laplace.ctl` codes `probit_b1.ferx` (2400
//! Bernoulli rows, `P(Y = 1) = Φ(TVL + PLACEBO[study, visit] + η + EMAX·TRT)`, ID = study)
//! with every level a `THETA` and `P = PHI(LP)`, under `METHOD=COND LAPLACE LIKE`. ferx is
//! handed NONMEM's whole optimum and evaluates once. ferx's FOCE and FOCEI on a binary row
//! are the Laplace form (#2437), so all three methods are held to the one NONMEM objective.
//! `probit_b1_tail.ctl` evaluates the same model with `MAXEVAL=0` at `TVL = −2.8`, where
//! `Φ(LP)` is about 3e-3 and the observed rate is about one half, so the probit term sits far
//! from its centre on most records.
//!
//! **Profiled (#2111 + #2603).** The `[binary_model]` predictor is already an implicitly
//! linked route, so a `profile`d block read by a probit predictor needs no route change:
//! ferx is handed NONMEM's structural point only and solves the 30 levels itself. A bounded
//! endpoint read through an outermost `pnorm(affine)` is the probit form of `y =
//! inv_logit(affine)`; its profiled and unprofiled objectives agree at the same point.

#![cfg(feature = "survival")]

use ferx_core::types::{EstimationMethod, FitOptions, FitResult, ModelParameters, OmegaMatrix};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The final row of `nonmem_anchor/results/<name>.ext`: the 32 θ, Ω and the objective.
fn nonmem_row(name: &str) -> (Vec<f64>, f64, f64) {
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
                let theta = (1..=32).map(|k| get(&format!("THETA{k}"))).collect();
                return (theta, get("OMEGA(1,1)"), get("OBJ"));
            }
            _ => {}
        }
    }
    panic!("{name}.ext: no final row");
}

/// The B1-probit model with `edit` applied to its text, loaded against its data.
fn prep(edit: impl Fn(String) -> String) -> ferx_core::PreparedRun {
    let text = std::fs::read_to_string(root().join("nonmem_anchor/probit_b1.ferx")).unwrap();
    let text = edit(text);
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("ferx_probit_b1_{}_{n}.ferx", std::process::id()));
    std::fs::write(&path, text).unwrap();
    let p = ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(root().join("nonmem_anchor/probit_b1.csv").to_str().unwrap()),
    )
    .unwrap_or_else(|e| panic!("B1-probit loads: {e}"));
    let _ = std::fs::remove_file(&path);
    p
}

fn evaluate_at(
    p: &ferx_core::PreparedRun,
    params: &ModelParameters,
    method: EstimationMethod,
    interaction: bool,
    covariance: bool,
) -> FitResult {
    let options = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: covariance,
        inner_tol: 1e-8,
        method,
        interaction,
        ..p.parsed.fit_options.clone()
    };
    ferx_core::fit(&p.parsed.model, &p.population, params, &options)
        .unwrap_or_else(|e| panic!("{method:?}: {e}"))
}

/// No warning may say a gradient or Hessian left the exact (dual) path.
fn assert_no_fd(what: &str, f: &FitResult) {
    for w in &f.warnings {
        assert!(
            !w.contains("finite difference") && !w.contains("finite-difference"),
            "{what}: {w}"
        );
    }
}

/// B1-probit, unprofiled, at the parameters of NONMEM's `name` run.
fn unprofiled_at(
    name: &str,
    method: EstimationMethod,
    interaction: bool,
    edit: impl Fn(String) -> String,
) -> (FitResult, f64) {
    let p = prep(edit);
    assert_eq!(p.parsed.model.theta_names.len(), 32);
    assert_eq!(p.parsed.model.theta_names[2], "PLACEBO[STUDY=1,TIME=0]");
    assert_eq!(p.parsed.model.theta_names[31], "PLACEBO[STUDY=10,TIME=8]");
    let (theta, omega, obj) = nonmem_row(name);
    let mut params = p.init_params.clone();
    params.theta = theta;
    params.omega = OmegaMatrix::from_diagonal(&[omega], vec!["ETA_S".into()]);
    (evaluate_at(&p, &params, method, interaction, false), obj)
}

/// The same objective at the same two points, computed outside ferx and NONMEM by
/// `nonmem_anchor/laplace_b1_check.py`: a hand-written Laplace per study with `Φ` from
/// `math.erfc` (correct to about an ulp). On the logit twin B1 the same script reproduces
/// NONMEM to −1.41e-9 (ferx: −1.42e-9), so it is the same objective; on probit it is the
/// reference that sees the link exactly.
const EXACT_PHI_OFV: [(&str, f64); 2] = [
    ("probit_b1_laplace", 3019.7507340385255),
    ("probit_b1_tail", 3882.8338559211643),
];

/// |ΔOFV| bound against the exact-`Φ` reference. Measured 1.36e-12 at NONMEM's optimum and
/// 2.27e-12 at the tail point, every method, macOS arm64 (Linux aarch64: 1.36e-12 / 2.73e-12);
/// the bound keeps 36× headroom.
/// Dropping the probit arm for the logit one moves the objective by 37.7.
const EXACT_OFV: f64 = 1e-10;

/// |ΔOFV| bound against NONMEM. **NONMEM's `PHI` is an approximation**: tabulated at known
/// points (`probit_phi_probe.ctl`, `results/probit_phi_probe.tab`) it is off by 1e-8 to 3e-7
/// relative over |x| ≤ 2, 2.3e-6 at −2.5, 5e-5 at −3 and 1e-2 at −8, against `erfc`. That,
/// not ferx, is the realised gap: ferx −2.13e-6 at NONMEM's optimum and +5.2e-6 low at the
/// tail point (`probit_b1_tail`, where most records sit near `Φ = 3e-3`), and the exact-`Φ`
/// reference shows the same two gaps to 1e-10. The bound keeps ~4× headroom over the larger;
/// the logit/probit swap moves the objective by 37.7, four decades further.
const NONMEM_OFV: f64 = 2e-5;

/// A-B1-probit under `method`: ferx at NONMEM's optimum (`probit_b1_laplace`) and at the tail
/// point (`probit_b1_tail`), against the exact-`Φ` reference and against NONMEM's OBJ, on the
/// exact discrete η-derivatives (no finite-difference warning).
fn assert_reproduces(method: EstimationMethod, interaction: bool) {
    for (name, exact) in EXACT_PHI_OFV {
        let (f, obj) = unprofiled_at(name, method, interaction, |t| t);
        let (de, dn) = (f.ofv - exact, f.ofv - obj);
        eprintln!(
            "A-B1-probit {name} {method:?}: ferx {:.10}, vs exact Φ {de:.3e}, vs NONMEM {dn:.3e}",
            f.ofv
        );
        assert!(f.ofv.is_finite(), "{name} {method:?}: OFV {}", f.ofv);
        assert!(
            de.abs() <= EXACT_OFV,
            "{name} {method:?}: vs exact Φ {de:.3e}"
        );
        assert!(
            dn.abs() <= NONMEM_OFV,
            "{name} {method:?}: vs NONMEM {dn:.3e}"
        );
        assert_no_fd(name, &f);
    }
}

#[test]
fn probit_binary_reproduces_nonmem_laplace() {
    assert_reproduces(EstimationMethod::Laplace, true);
}

#[test]
fn probit_binary_reproduces_nonmem_under_focei() {
    assert_reproduces(EstimationMethod::FoceI, true);
}

#[test]
fn probit_binary_reproduces_nonmem_under_foce() {
    assert_reproduces(EstimationMethod::Foce, false);
}

/// The anchor sees the link: the same point scored under the logit link is far from NONMEM's
/// probit objective, so a probit arm that scored logit could not pass the tests above. The
/// pair straddles the link: identical file, data and point, one word changed.
#[test]
fn the_anchor_tells_probit_from_logit() {
    let to_logit = |t: String| {
        assert!(
            t.contains("  link   = probit\n  probit = "),
            "the fixture moved"
        );
        t.replace("  link   = probit\n  probit = ", "  logit = ")
    };
    let (f, obj) = unprofiled_at(
        "probit_b1_laplace",
        EstimationMethod::Laplace,
        true,
        to_logit,
    );
    eprintln!("B1-probit scored under logit: ΔOFV {:.3}", f.ofv - obj);
    assert!((f.ofv - obj).abs() > 1.0, "{} vs {obj}", f.ofv);
}

// ── Profiled ─────────────────────────────────────────────────────────────────────

const UNPROFILED: &str = "](0.0, -10.0, 10.0)\n";
const PROFILED: &str = "](0.0) profile\n";

fn profile(t: String) -> String {
    assert!(t.contains(UNPROFILED), "the fixture moved");
    t.replace(UNPROFILED, PROFILED)
}

/// PR-bin: ferx handed NONMEM's structural point (TVL, EMAX, ω) solves the 30 free levels
/// itself, through the probit predictor, under each method. The block is reported profiled
/// and nothing falls back to finite differences. The solved objective is the minimum over the
/// levels, so it is at or below the exact-`Φ` objective at NONMEM's levels (which optimise
/// NONMEM's approximate `PHI`), and only just below; it is within [`NONMEM_OFV`] of NONMEM's
/// OBJ, and the levels are NONMEM's to 1e-4. Measured (macOS arm64), every method: −3.38e-8
/// below the exact-`Φ` objective at NONMEM's levels, −2.16e-6 from NONMEM, levels within
/// 1.24e-5.
fn assert_profiled_reproduces(method: EstimationMethod, interaction: bool) {
    let (theta, omega, obj) = nonmem_row("probit_b1_laplace");
    let p = prep(profile);
    let mut params = p.init_params.clone();
    params.theta[0] = theta[0];
    params.theta[1] = theta[1];
    params.omega = OmegaMatrix::from_diagonal(&[omega], vec!["ETA_S".into()]);
    let f = evaluate_at(&p, &params, method, interaction, false);
    assert_eq!(f.profiled_blocks, ["PLACEBO"], "{method:?}");
    assert_no_fd("profiled", &f);
    let d = f.ofv - obj;
    let below = f.ofv - EXACT_PHI_OFV[0].1;
    let mut level = 0.0f64;
    for k in 2..32 {
        let g = (f.theta[k] - theta[k]).abs();
        assert!(g.is_finite(), "{method:?} θ{k}");
        level = level.max(g);
    }
    eprintln!(
        "A-B1-probit profiled {method:?}: ferx {:.10}, vs NONMEM {d:.3e}, vs exact Φ at \
         NONMEM's levels {below:.3e}, levels {level:.3e}",
        f.ofv
    );
    assert!(
        below <= 1e-9 && below > -1e-6,
        "{method:?}: vs exact Φ {below:.3e}"
    );
    assert!(d.abs() <= NONMEM_OFV, "{method:?}: vs NONMEM {d:.3e}");
    assert!(level < 1e-4, "{method:?}: levels {level:.3e}");
}

#[test]
fn profiled_probit_levels_reproduce_nonmem_laplace() {
    assert_profiled_reproduces(EstimationMethod::Laplace, true);
}

#[test]
fn profiled_probit_levels_reproduce_nonmem_under_focei() {
    assert_profiled_reproduces(EstimationMethod::FoceI, true);
}

#[test]
fn profiled_probit_levels_reproduce_nonmem_under_foce() {
    assert_profiled_reproduces(EstimationMethod::Foce, false);
}

/// The profiled `R` against the unprofiled step at the profiled point, every level a free θ,
/// under `laplace`, as B1-logit's C1-bin. Measured: ΔOFV 9.1e-13, worst SE 4.4e-15 relative
/// (macOS arm64).
#[test]
fn the_profiled_probit_covariance_is_the_unprofiled_one_at_the_same_point() {
    let (theta, omega, _) = nonmem_row("probit_b1_laplace");
    let pp = prep(profile);
    let mut params = pp.init_params.clone();
    params.theta[0] = theta[0];
    params.theta[1] = theta[1];
    params.omega = OmegaMatrix::from_diagonal(&[omega], vec!["ETA_S".into()]);
    let fp = evaluate_at(&pp, &params, EstimationMethod::Laplace, true, true);
    let pu = prep(|t| t);
    let mut params = pu.init_params.clone();
    params.theta.clone_from(&fp.theta);
    params.omega = OmegaMatrix::from_diagonal(&[omega], vec!["ETA_S".into()]);
    let fu = evaluate_at(&pu, &params, EstimationMethod::Laplace, true, true);
    let sp = fp
        .se_theta
        .as_ref()
        .unwrap_or_else(|| panic!("profiled covariance: {:?}", fp.warnings));
    let su = fu
        .se_theta
        .as_ref()
        .unwrap_or_else(|| panic!("unprofiled covariance: {:?}", fu.warnings));
    assert_eq!(sp.len(), su.len());
    let mut worst = 0.0_f64;
    for (a, b) in sp.iter().zip(su) {
        let rel = ((a - b) / b).abs();
        assert!(rel.is_finite(), "{a} vs {b}");
        worst = worst.max(rel);
    }
    eprintln!(
        "C1-probit: ΔOFV {:.3e}, worst SE {worst:.3e}",
        fu.ofv - fp.ofv
    );
    assert!((fu.ofv - fp.ofv).abs() < 1e-7);
    assert!(worst < 1e-9, "{worst:.3e}");
}

// ── A bounded endpoint through `pnorm` ───────────────────────────────────────────

/// `profile_logit_b2.ferx` (observed proportions, Gaussian rows) with its logit link
/// replaced by `pnorm`, profiled or not.
fn b2_pnorm(profiled: bool) -> ferx_core::PreparedRun {
    let text = std::fs::read_to_string(root().join("nonmem_anchor/profile_logit_b2.ferx")).unwrap();
    assert!(text.contains("y = inv_logit(LP)"), "the fixture moved");
    assert!(text.contains("](0.0) profile\n"), "the fixture moved");
    let mut text = text.replace("y = inv_logit(LP)", "y = pnorm(LP)");
    if !profiled {
        text = text.replace("](0.0) profile\n", "](0.0, -10.0, 10.0)\n");
    }
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("ferx_probit_b2_{}_{n}.ferx", std::process::id()));
    std::fs::write(&path, text).unwrap();
    let p = ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(
            root()
                .join("nonmem_anchor/profile_logit_b2.csv")
                .to_str()
                .unwrap(),
        ),
    )
    .unwrap_or_else(|e| panic!("B2-pnorm loads: {e}"));
    let _ = std::fs::remove_file(&path);
    p
}

/// PR-bnd: a block read through an outermost `pnorm(affine)` profiles like one read through
/// `inv_logit(affine)`. At the model's start, the profiled evaluation solves the levels, and
/// the unprofiled model evaluated at the solved levels gives the same objective — the solve's
/// minimum is a point of the unprofiled objective — under `focei` and `laplace`. The block is
/// reported profiled and nothing falls back to finite differences. A solve that stopped
/// short, or scored a different objective, opens a gap. Measured: 2.0e-11 (`focei`) and
/// 2.0e-11 (`laplace`), macOS arm64.
#[test]
fn a_block_read_through_pnorm_profiles_to_the_unprofiled_objective() {
    for (method, interaction) in [
        (EstimationMethod::FoceI, true),
        (EstimationMethod::Laplace, true),
    ] {
        let pp = b2_pnorm(true);
        let fp = evaluate_at(&pp, &pp.init_params, method, interaction, false);
        assert_eq!(fp.profiled_blocks, ["PLACEBO"], "{method:?}");
        assert_no_fd("pnorm profiled", &fp);
        let pu = b2_pnorm(false);
        let mut params = pu.init_params.clone();
        params.theta.clone_from(&fp.theta);
        let fu = evaluate_at(&pu, &params, method, interaction, false);
        assert!(fu.profiled_blocks.is_empty());
        let d = fu.ofv - fp.ofv;
        eprintln!(
            "B2-pnorm {method:?}: profiled {:.10} unprofiled {:.10} Δ {d:.3e}",
            fp.ofv, fu.ofv
        );
        assert!(fp.ofv.is_finite() && fu.ofv.is_finite());
        assert!(d.abs() < 1e-7, "{method:?}: Δ {d:.3e}");
        // The levels moved off their start: the solve ran.
        let moved = fp.theta[2..].iter().any(|v| v.abs() > 1e-3);
        assert!(moved, "{method:?}: the levels stayed at 0");
    }
}
