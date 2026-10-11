//! NONMEM anchor for the per-level standard errors of a θ level block (#1890).
//!
//! A dependent level of `sum_to_zero_within` has no θ, so NONMEM has no direct
//! spelling of its standard error. It has one by **reparameterisation**: code the same
//! contrast with a different level dependent, and the level ferx derives becomes a
//! NONMEM `THETA` with its own `$COV` standard error. Two control streams, one model
//! (`nonmem_anchor/level_se_mbma_fit.ferx`, `mbma_placebo` without the between-arm
//! kappa):
//!
//! - `level_se_a.ctl`: the **last** visit of each study dependent, ferx's own layout.
//!   Its free levels are the comparator for ferx's free-level SEs, and its optimum is
//!   where ferx is evaluated (`outer_maxiter = 0` plus the covariance step: an
//!   evaluation, not a fit, so Tier 2).
//! - `level_se_b.ctl`: the **first** visit dependent, so each level ferx derives is
//!   estimated directly. Its `THETA` SEs are the comparator for ferx's dependent-level
//!   SEs.
//!
//! Both converge to the same objective (165.826023 / 165.826024, `nm3` anchor image)
//! and agree with each other: the delta method on (a)'s `.cov` against (b)'s `$COV`
//! SE differs by at most 5.5e-5 relative — NONMEM's own floor here, set by its
//! three-significant-digit convergence and its finite-difference R matrix.
//!
//! The comparison is non-degenerate: every study has its dependent level read through
//! three free θ with a nonzero covariance block, the six dependent values differ
//! (−2.48 … −0.01), and the free-level comparison runs on the same matrix, so a gap
//! in the dependent levels can be told from a gap in the covariance step itself.
//!
//! What it does not see: the level θ here are bounded below by −50, so they are
//! identity-packed and the packing Jacobian `d` is 1 throughout. The log-packed
//! dependent path (`d = θ`) is pinned only inside ferx — by U3's log-packed arm (an FD
//! of the gather's value in packed coordinates) and U2's bit-parity with `se_theta`
//! in `src/api/tests/theta_levels_tests.rs` — not by NONMEM.

use ferx_core::types::{FitOptions, OmegaMatrix};
use std::collections::HashMap;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The final estimates (`ITERATION = -1000000000`) and standard errors
/// (`-1000000001`) of a NONMEM `.ext` file, by column name.
fn ext(name: &str) -> (HashMap<String, f64>, HashMap<String, f64>) {
    let text = std::fs::read_to_string(root().join("nonmem_anchor/results").join(name)).unwrap();
    let mut names: Vec<String> = Vec::new();
    let (mut est, mut se) = (HashMap::new(), HashMap::new());
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.first() {
            Some(&"TABLE") | None => continue,
            Some(&"ITERATION") => names = fields.iter().map(|s| s.to_string()).collect(),
            Some(it) => {
                let row: HashMap<String, f64> = names
                    .iter()
                    .cloned()
                    .zip(fields.iter().map(|v| v.parse::<f64>().unwrap()))
                    .collect();
                match *it {
                    "-1000000000" => est = row,
                    "-1000000001" => se = row,
                    _ => {}
                }
            }
        }
    }
    assert!(
        !est.is_empty() && !se.is_empty(),
        "{name}: no final estimates or SEs"
    );
    (est, se)
}

#[test]
fn level_standard_errors_match_nonmem_under_both_parameterisations() {
    let (est_a, se_a) = ext("level_se_a.ext");
    let (_, se_b) = ext("level_se_b.ext");

    let prep = ferx_core::prepare_run(
        root()
            .join("nonmem_anchor/level_se_mbma_fit.ferx")
            .to_str()
            .unwrap(),
        Some(
            root()
                .join("tests/data/mbma_placebo/mbma_placebo.csv")
                .to_str()
                .unwrap(),
        ),
    )
    .expect("prepare");
    let model = &prep.parsed.model;
    assert_eq!(model.n_theta, 22, "4 structural θ and 18 free levels");

    // NONMEM (a)'s THETA k is ferx's θ[k − 1]: the structural four, then each study's
    // first three visits in level order, the fourth dependent — the same layout.
    let mut params = prep.init_params.clone();
    params.theta = (1..=22).map(|k| est_a[&format!("THETA{k}")]).collect();
    params.omega = OmegaMatrix::from_diagonal(&[est_a["OMEGA(1,1)"]], vec!["ETA_E0".into()]);
    let opts = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: true,
        ..prep.parsed.fit_options.clone()
    };
    let f = ferx_core::fit(model, &prep.population, &params, &opts).expect("evaluation");
    assert!(
        (f.ofv - est_a["OBJ"]).abs() < 1e-3,
        "OFV at NONMEM's optimum: ferx {} vs NONMEM {}",
        f.ofv,
        est_a["OBJ"]
    );
    let cov = f.covariance_matrix.as_ref().expect("covariance step");
    let map = ferx_core::theta_level_values(model, &f.theta, Some(cov), None).unwrap();
    let levels = &map["PLACEBO"];
    assert_eq!(levels.len(), 24);

    let (mut worst_free, mut worst_dep, mut n_dep) = (0.0f64, 0.0f64, 0);
    for (i, l) in levels.iter().enumerate() {
        let (study, visit) = (i / 4, i % 4);
        let got = l.se.unwrap_or_else(|| panic!("{}: no SE", l.label));
        let want = match l.theta_index {
            // A free level: NONMEM (a)'s own THETA, the same index plus one.
            Some(k) => se_a[&format!("THETA{}", k + 1)],
            // The fourth visit, dependent in ferx and in (a): in (b) the free θ of
            // study s are visits 2, 3, 4, at THETA 5 + 3s, 6 + 3s, 7 + 3s.
            None => {
                assert_eq!(visit, 3, "{}: ferx derives the last visit", l.label);
                n_dep += 1;
                se_b[&format!("THETA{}", 7 + 3 * study)]
            }
        };
        assert!(
            got.is_finite() && want.is_finite(),
            "{}: {got} vs {want}",
            l.label
        );
        let rel = (got - want).abs() / want;
        if l.theta_index.is_some() {
            worst_free = worst_free.max(rel);
        } else {
            worst_dep = worst_dep.max(rel);
        }
        eprintln!(
            "{:20} ferx {got:.6} NONMEM {want:.6} rel {rel:.2e}",
            l.label
        );
    }
    eprintln!("worst relative SE gap: free {worst_free:.2e}, dependent {worst_dep:.2e}");
    assert_eq!(n_dep, 6);
    assert!(worst_free < BOUND, "free levels: {worst_free:e}");
    assert!(worst_dep < BOUND, "dependent levels: {worst_dep:e}");
}

/// Measured on Linux aarch64 (`tools/linux-test.sh`, glibc 2.41) and on macOS arm64,
/// identical in every printed digit: worst relative SE gap 3.44e-3 over the 18 free
/// levels and 4.22e-3 over the 6 dependent ones. The two are the same size, so
/// the dependent-level rule adds nothing measurable to the gap the covariance step
/// itself has against NONMEM (its FD R matrix against NONMEM's, at a point printed to
/// six digits). The bound is 2.4× the worst. Reading the diagonal only — dropping the
/// group's covariances — misses by 0.70 (measured with that mutation), far outside it.
const BOUND: f64 = 1e-2;
