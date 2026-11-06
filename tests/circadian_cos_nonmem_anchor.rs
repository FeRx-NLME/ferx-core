//! NONMEM `$PRED` anchor for `sin(...)` / `cos(...)` (#2720): a circadian baseline
//! with the phase random effect inside both trigonometric arguments.
//!
//! ```text
//! CIRC = AMP1·cos(2π(TIME + SHIFT)/24 − PHS1) + AMP2·sin(4π(TIME + SHIFT)/24 − PHS2)
//! y    = BASE·(1 + CIRC)·(1 + ε),  BASE = TVBASE·e^{η₁},  AMP1 = TVAMP1·e^{η₂},  SHIFT = η₃
//! ```
//!
//! The kit (`nonmem_anchor/`):
//!
//! - `circadian_cos.csv` — 30 subjects × 10 samples over 1.5 days, start times staggered
//!   per subject, simulated by `simulate_circadian_cos.py` (seed 2720);
//! - `circadian_cos.ctl` — NONMEM's `COS`/`SIN`, FOCE-I at `MAXEVAL=0` with `POSTHOC`,
//!   evaluated at initial estimates off the simulation truth, `FORMAT=s1PE23.16`;
//! - `circadian_cos_est.ctl` — the same model fitted to convergence, `$COV MATRIX=R`;
//! - `circadian_cos_cov.ctl` — NONMEM's R-matrix alone at that optimum, with the objective
//!   and the `$COV` differences run tight (`SIGL=14`, `SIGL=12 TOL=12`); the SE reference;
//! - `circadian_cos_fit.ferx` — the ferx twin, spelled with the DSL's `cos`/`sin`, at the
//!   same initial estimates.
//!
//! Results in `results/circadian_cos{,_est,_cov}.*`, NONMEM 7.6.0 on Linux x86_64.
//!
//! Non-degenerate: the phase η is inside **both** functions, so η̂₃ and the FOCE-I
//! objective's `∂f/∂η₃` run through `cos` and `sin`; the amplitude carries its own η, so
//! a wrong `cos` value moves IPRED by a subject-specific factor rather than a common one;
//! `PHS1`, `PHS2` and every η̂₃ are nonzero (η̂₃ spans −6.4 … +3.3 h), and the staggered
//! sampling puts every subject's records at different points of the cycle. Swapping
//! `cos`/`sin` in either harmonic (a quarter-period phase error) moves IPRED at NONMEM's
//! η̂ by up to 36% (first harmonic) / 8% (second), computed outside ferx — far beyond
//! every bound below.

use ferx_core::types::FitOptions;
use std::collections::HashMap;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn results(name: &str) -> String {
    std::fs::read_to_string(root().join("nonmem_anchor/results").join(name)).unwrap()
}

/// The `.ext` row whose iteration is `row` (e.g. `-1000000000`), by column name.
fn ext(file: &str, row: &str) -> HashMap<String, f64> {
    let text = results(file);
    let mut names: Vec<String> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first() {
            Some(&"ITERATION") => names = f.iter().map(|s| s.to_string()).collect(),
            Some(r) if *r == row => {
                return names
                    .iter()
                    .cloned()
                    .zip(f.iter().map(|v| v.parse::<f64>().unwrap()))
                    .collect()
            }
            _ => {}
        }
    }
    panic!("no {row} row in {file}");
}

/// Per subject, NONMEM's `(ETA(1), ETA(2), ETA(3))` from the `.phi`.
fn phi(file: &str) -> Vec<[f64; 3]> {
    results(file)
        .lines()
        .skip(2)
        .map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .skip(2)
                .take(3)
                .map(|v| v.parse().unwrap())
                .collect();
            [f[0], f[1], f[2]]
        })
        .collect()
}

/// `(PRED, IPRED)` per record, in data order, from the `$TABLE`.
fn tab(file: &str) -> Vec<(f64, f64)> {
    let text = results(file);
    let mut lines = text.lines().skip(1);
    let header: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    let at = |n: &str| header.iter().position(|h| *h == n).unwrap();
    let (p, i) = (at("PRED"), at("IPRED"));
    lines
        .map(|l| {
            let f: Vec<f64> = l.split_whitespace().map(|v| v.parse().unwrap()).collect();
            (f[p], f[i])
        })
        .collect()
}

fn prepare() -> ferx_core::PreparedRun {
    ferx_core::prepare_run(
        root()
            .join("nonmem_anchor/circadian_cos_fit.ferx")
            .to_str()
            .unwrap(),
        Some(
            root()
                .join("nonmem_anchor/circadian_cos.csv")
                .to_str()
                .unwrap(),
        ),
    )
    .expect("the circadian anchor model must parse and bind")
}

/// ferx evaluated (`outer_maxiter = 0`) at NONMEM's `MAXEVAL=0` point: objective, each
/// subject's three η̂, and PRED / IPRED at every record.
#[test]
fn circadian_cos_sin_evaluation_matches_nonmem() {
    let est = ext("circadian_cos.ext", "-1000000000");
    let prep = prepare();
    assert!(
        prep.parsed.model.is_algebraic(),
        "the anchor is the $PRED path"
    );
    let params = prep.init_params.clone();
    // The .ferx initial estimates are the control stream's, which NONMEM echoes.
    let theta_nm: Vec<f64> = (1..=5).map(|k| est[&format!("THETA{k}")]).collect();
    assert_eq!(params.theta, theta_nm, "θ");
    for (k, name) in [(0, "OMEGA(1,1)"), (1, "OMEGA(2,2)"), (2, "OMEGA(3,3)")] {
        assert_eq!(params.omega.matrix[(k, k)], est[name], "{name}");
    }
    assert!(
        (params.sigma.values[0].powi(2) - est["SIGMA(1,1)"]).abs() < 1e-15,
        "σ² {} vs {}",
        params.sigma.values[0].powi(2),
        est["SIGMA(1,1)"]
    );

    let opts = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..prep.parsed.fit_options.clone()
    };
    let f = ferx_core::fit(&prep.parsed.model, &prep.population, &params, &opts)
        .expect("evaluation at fixed parameters");

    let (etas, rows) = (phi("circadian_cos.phi"), tab("circadian_cos.tab"));
    assert_eq!(f.subjects.len(), 30);
    assert_eq!(etas.len(), f.subjects.len(), "subjects");

    let d_ofv = (f.ofv - est["OBJ"]).abs();
    assert!(
        d_ofv.is_finite(),
        "OFV: ferx {} NONMEM {}",
        f.ofv,
        est["OBJ"]
    );

    // The phase η̂ must be live, or the anchor does not see `cos`'s argument move.
    let shift_span = etas.iter().map(|e| e[2].abs()).fold(0.0, f64::max); // nan-ok: parsed from the .phi, finite or the parse panics
    assert!(shift_span > 1.0, "η̂₃ span {shift_span}");

    let mut d_eta = 0.0f64;
    for (s, want) in f.subjects.iter().zip(&etas) {
        for k in 0..3 {
            let d = (s.eta[k] - want[k]).abs();
            assert!(d.is_finite(), "{}: η̂{k} {} vs {}", s.id, s.eta[k], want[k]);
            d_eta = d_eta.max(d);
        }
    }
    let preds: Vec<(f64, f64)> = f
        .subjects
        .iter()
        .flat_map(|s| s.pred.iter().copied().zip(s.ipred.iter().copied()))
        .collect();
    assert_eq!(preds.len(), rows.len(), "records");
    let (mut d_pred, mut d_ipred) = (0.0f64, 0.0f64);
    for ((p, i), (wp, wi)) in preds.iter().zip(&rows) {
        let (dp, di) = ((p - wp).abs() / wp.abs(), (i - wi).abs() / wi.abs());
        assert!(
            dp.is_finite() && di.is_finite(),
            "PRED {p}/{wp}, IPRED {i}/{wi}"
        );
        d_pred = d_pred.max(dp);
        d_ipred = d_ipred.max(di);
    }
    eprintln!(
        "ferx vs NONMEM: |ΔOFV| {d_ofv:.3e} (ferx {:.10}, NONMEM {:.10}), max |Δη̂| \
         {d_eta:.3e}, max rel ΔPRED {d_pred:.3e}, max rel ΔIPRED {d_ipred:.3e}",
        f.ofv, est["OBJ"]
    );
    // Realised (Linux x86_64 debug) are in the bounds' comments. PRED is a closed form
    // at η = 0 on both sides, printed by NONMEM to 17 digits; η̂ comes from two different
    // inner optimizers and the `.phi` prints 6 significant digits, so that bound is the
    // print precision of η̂₃ ≈ 4 (5e-5) with headroom; IPRED inherits η̂'s error.
    assert!(d_pred < 1e-13, "PRED: {d_pred:e}");
    assert!(d_eta < 2e-4, "η̂: {d_eta:e}");
    assert!(d_ipred < 1e-5, "IPRED: {d_ipred:e}");
    assert!(d_ofv < 1e-4, "OFV: {d_ofv:e}");
}

/// Both engines fitted to convergence from the same initial estimates: OFV, θ, Ω, σ and
/// the R-matrix SEs. The SEs exercise the second-order blocks of the `sin`/`cos` jet.
///
/// The SE reference is `circadian_cos_cov.ctl`, not the estimation run's own `$COV`. At
/// NONMEM's default precision that put SE(PHS1) and SE(PHS2) ~30% and SE(Ω₁₁), SE(σ)
/// 14–18% above ferx's, while ferx's exact R-matrix and its FD-Hessian fallback
/// (`analytic_cov_hessian = false`) agreed with each other to ~1e-3 — and both engines'
/// objectives agree to 6e-8 at a common point (the test above). Rerun tight at the
/// optimum, NONMEM moved onto ferx on every element, so the gap was NONMEM's numerical
/// R-matrix, measured, not assumed.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: FOCE-I fit to convergence (#2720): opt in with --features slow-tests"
)]
fn circadian_cos_sin_fit_matches_nonmem() {
    let est = ext("circadian_cos_est.ext", "-1000000000");
    let se = ext("circadian_cos_cov.ext", "-1000000001");
    let se_sd = ext("circadian_cos_cov.ext", "-1000000005");
    let prep = prepare();
    let opts = FitOptions {
        verbose: false,
        run_covariance_step: true,
        ..prep.parsed.fit_options.clone()
    };
    let f = ferx_core::fit(
        &prep.parsed.model,
        &prep.population,
        &prep.init_params,
        &opts,
    )
    .expect("the circadian fit must converge");

    let d_ofv = (f.ofv - est["OBJ"]).abs();
    eprintln!("OFV: ferx {:.6} NONMEM {:.6}", f.ofv, est["OBJ"]);
    let rel = |got: f64, want: f64, what: &str| {
        let r = (got - want).abs() / want.abs();
        eprintln!("{what}: ferx {got:.6e} NONMEM {want:.6e} rel {r:.2e}");
        r
    };
    let (mut ests, mut ses) = (Vec::new(), Vec::new());
    let se_theta = f.se_theta.as_ref().expect("θ SEs");
    for k in 0..5 {
        let name = format!("THETA{}", k + 1);
        ests.push(rel(f.theta[k], est[&name], &name));
        ses.push(rel(se_theta[k], se[&name], &format!("SE {name}")));
    }
    let se_omega = f.se_omega.as_ref().expect("Ω SEs");
    for (k, name) in ["OMEGA(1,1)", "OMEGA(2,2)", "OMEGA(3,3)"]
        .iter()
        .enumerate()
    {
        ests.push(rel(f.omega[(k, k)], est[*name], name));
        ses.push(rel(se_omega[k], se[*name], &format!("SE {name}")));
    }
    // NONMEM's SIGMA is a variance; ferx reports the proportional SD and its SE.
    ests.push(rel(f.sigma[0], est["SIGMA(1,1)"].sqrt(), "σ (SD)"));
    ses.push(rel(
        f.se_sigma.as_ref().expect("σ SE")[0],
        se_sd["SIGMA(1,1)"],
        "SE σ (SD)",
    ));
    assert!(ests.iter().all(|e| e.is_finite()), "estimates: {ests:?}");
    assert!(ses.iter().all(|e| e.is_finite()), "SEs: {ses:?}");
    let worst_est = ests.iter().fold(0.0, |w: f64, &e| w.max(e));
    let worst_se = ses.iter().fold(0.0, |w: f64, &e| w.max(e));
    eprintln!("worst relative: estimates {worst_est:.2e}, SEs {worst_se:.2e}");
    // NONMEM stops with ROUNDING ERRORS (ERROR=134), as on the other simulated $PRED
    // anchors, at the minimum (its `.ext` prints 6 significant digits); evaluated at
    // ferx's optimum its OFV is 1386.393037, below its own final 1386.393044.
    // Realised (Linux x86_64): |ΔOFV| 6e-6, estimates 5.5e-4 (Ω₁₁; θ ≤ 3.1e-4), SEs 3.1e-2
    // (PHS2; PHS1 3.0e-2, everything else ≤ 1.2e-2). Bounds ~2× the realised.
    assert!(d_ofv < 1e-4, "OFV: {d_ofv:e}");
    assert!(worst_est < 1e-3, "estimates: {worst_est:e}");
    assert!(worst_se < 6e-2, "SEs: {worst_se:e}");
}
