//! NONMEM anchors for #1569: a transit (`transit_iov_mtt`) and an inverse-Gaussian
//! (`ig_iov_mat`) closed-form model under IOV on `CL` **and** on the absorption parameter
//! (`MTT` / `MAT`), with doses still absorbing when their occasion's successor begins.
//!
//! ferx serves both with the exact absorption walk (#1560), and a subject outside the walk
//! with the ODE twin; since #1569 both absorb each dose through the kernel of **its own dose
//! record**. The control streams
//! (`nonmem_anchor/{transit_iov_mtt,ig_iov_mat}.ctl`) code the same rule in `$DES`: each dose's
//! occasion is captured at its dose record and its `MTT`/`MAT` rebuilt from that occasion's
//! ETA, and `$DES` superposes all four doses. The disposition is the current record's in both
//! engines (NONMEM end-of-interval, #1073).
//!
//! Data (`simulate_absorption_iov_carryover_data.py`): 24 subjects, doses of 100 at 0/6/12/18 h;
//! occasion 2 begins at the **observation** record `t = 8.5` while dose 2 is ~85 % unabsorbed,
//! occasion 3 at the **dose** record `t = 18` while dose 3 is ~15 % unabsorbed. NONMEM 7.6.0
//! FOCEI (`ADVAN13 TOL=12`, `nonmem:7.6.0-anchor`) estimates from the simulation truth; ferx is
//! evaluated at NONMEM's optimum, read at full precision (`FORMAT=s1PE23.16`) from the committed
//! `nonmem_anchor/results/*.{ext,phi,tab}`.
//!
//! Two checks per anchor, each Tier 2 (one evaluation, no convergence loop):
//!
//! * **IPRED at NONMEM's EBEs** — ferx's `predict_iov` at NONMEM's POSTHOC η/κ against the
//!   `$TABLE` IPRED, every observation. This is the direct test of the kernel rule, and it
//!   runs on every PR, on **both engines**: the plain model runs on the walk, and a copy whose
//!   `CL` reads `TIME` through the inert factor `(1 + 0·TIME)` runs on the twin (a `TIME` read
//!   is outside the walk). The two legs must not be bit-identical, so neither can quietly
//!   take the other's engine; `pk::absorption_walk::tests` pins the walk route on this data.
//! * **FOCEI objective at NONMEM's optimum** — `fit()` with every parameter `FIX`, against
//!   NONMEM's objective, with per-subject objective contributions against `.phi`. This one
//!   runs nightly: see its doc comment for the measured CI cost and the per-PR tests that
//!   die on the same mutations.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::collections::HashMap;
use std::path::Path;

struct Anchor {
    /// `nonmem_anchor/<name>.{ctl,csv}`, `nonmem_anchor/results/<name>.*`.
    name: &'static str,
    /// The closed-form structural line; the kernel's second parameter is fixed (`TVB`).
    structural: &'static str,
    /// `[individual_parameters]` lines for the kernel: `ABS` carries the occasion κ.
    kernel_params: &'static str,
}

const ANCHORS: [Anchor; 2] = [
    Anchor {
        name: "transit_iov_mtt",
        structural: "pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)",
        kernel_params: "  MTT = TVA * exp(KAPPA_ABS)\n  NTR = TVB",
    },
    Anchor {
        name: "ig_iov_mat",
        structural: "pk one_cpt_ig(cl=CL, v=V, mat=MAT, cv2=CV2)",
        kernel_params: "  MAT = TVA * exp(KAPPA_ABS)\n  CV2 = TVB",
    },
];

/// NONMEM's final estimates, from the `.ext` row `ITERATION = -1000000000`.
struct Estimates {
    /// `[TVCL, TVV, TVA, TVB]`.
    theta: [f64; 4],
    sigma_var: f64,
    omega_cl: f64,
    omega_v: f64,
    omega_iov_cl: f64,
    omega_iov_abs: f64,
    ofv: f64,
}

fn whitespace_rows(path: &str) -> Vec<Vec<String>> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .lines()
        .map(|l| l.split_whitespace().map(str::to_string).collect())
        .collect()
}

fn estimates(name: &str) -> Estimates {
    let rows = whitespace_rows(&format!("nonmem_anchor/results/{name}.ext"));
    let header = rows
        .iter()
        .find(|r| r.first().map(String::as_str) == Some("ITERATION"))
        .expect(".ext header");
    let last = rows
        .iter()
        .find(|r| r.first().map(String::as_str) == Some("-1000000000"))
        .expect(".ext final-estimate row");
    let col = |label: &str| -> f64 {
        let i = header
            .iter()
            .position(|h| h == label)
            .unwrap_or_else(|| panic!(".ext has no {label}"));
        last[i].parse().expect(".ext value")
    };
    Estimates {
        theta: [col("THETA1"), col("THETA2"), col("THETA3"), col("THETA4")],
        sigma_var: col("SIGMA(1,1)"),
        omega_cl: col("OMEGA(1,1)"),
        omega_v: col("OMEGA(2,2)"),
        omega_iov_cl: col("OMEGA(3,3)"),
        omega_iov_abs: col("OMEGA(6,6)"),
        ofv: col("OBJ"),
    }
}

/// Per subject: NONMEM's POSTHOC `ETA(1..8)` and its objective contribution.
fn phi(name: &str) -> HashMap<String, ([f64; 8], f64)> {
    let rows = whitespace_rows(&format!("nonmem_anchor/results/{name}.phi"));
    let header = rows
        .iter()
        .find(|r| r.first().map(String::as_str) == Some("SUBJECT_NO"))
        .expect(".phi header");
    let at = |label: &str| header.iter().position(|h| h == label).expect(label);
    let eta_cols: Vec<usize> = (1..=8).map(|k| at(&format!("ETA({k})"))).collect();
    let (id_col, obj_col) = (at("ID"), at("OBJ"));
    let mut out = HashMap::new();
    for r in rows
        .iter()
        .filter(|r| r.len() == header.len() && r[0] != "SUBJECT_NO")
    {
        let id: f64 = r[id_col].parse().expect("ID");
        let mut eta = [0.0; 8];
        for (k, &c) in eta_cols.iter().enumerate() {
            eta[k] = r[c].parse().expect("ETA");
        }
        out.insert(
            format!("{}", id as i64),
            (eta, r[obj_col].parse().expect("OBJ")),
        );
    }
    out
}

/// Per subject, in data order: NONMEM's IPRED at each observation (`MDV = 0`).
fn table_ipred(name: &str) -> HashMap<String, Vec<f64>> {
    let rows = whitespace_rows(&format!("nonmem_anchor/results/{name}.tab"));
    let header = rows
        .iter()
        .find(|r| r.first().map(String::as_str) == Some("ID"))
        .expect(".tab header");
    let at = |label: &str| header.iter().position(|h| h == label).expect(label);
    let (id_col, mdv_col, ipred_col) = (at("ID"), at("MDV"), at("IPRED"));
    let mut out: HashMap<String, Vec<f64>> = HashMap::new();
    for r in rows
        .iter()
        .filter(|r| r.len() == header.len() && r[0] != "ID")
    {
        let mdv: f64 = r[mdv_col].parse().expect("MDV");
        if mdv != 0.0 {
            continue;
        }
        let id: f64 = r[id_col].parse().expect("ID");
        out.entry(format!("{}", id as i64))
            .or_default()
            .push(r[ipred_col].parse().expect("IPRED"));
    }
    out
}

/// The ferx model at NONMEM's optimum, every parameter `FIX`. `twin` adds the numerically
/// inert `(1 + 0·TIME)` to `CL`, which routes every subject to the ODE twin (#1560).
fn model_at(anchor: &Anchor, est: &Estimates, twin: bool) -> ferx_core::types::CompiledModel {
    let [tvcl, tvv, tva, tvb] = est.theta;
    let time_read = if twin { " * (1 + 0 * TIME)" } else { "" };
    let src = format!(
        r"
[parameters]
  theta TVCL({tvcl:e}, FIX)
  theta TVV({tvv:e}, FIX)
  theta TVA({tva:e}, FIX)
  theta TVB({tvb:e}, FIX)
  omega ETA_CL ~ {ocl:e} FIX
  omega ETA_V  ~ {ov:e} FIX
  kappa KAPPA_CL  ~ {oicl:e} FIX
  kappa KAPPA_ABS ~ {oiabs:e} FIX
  sigma PROP_ERR ~ {sd:e} (sd) FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL){time_read}
  V  = TVV  * exp(ETA_V)
{kernel}

[structural_model]
  {structural}

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  iov_column = OCC
  ode_reltol = 1e-12
  ode_abstol = 1e-12
",
        ocl = est.omega_cl,
        ov = est.omega_v,
        oicl = est.omega_iov_cl,
        oiabs = est.omega_iov_abs,
        sd = est.sigma_var.sqrt(),
        kernel = anchor.kernel_params,
        structural = anchor.structural,
    );
    parse_model_string(&src).expect("the anchor model parses")
}

fn population(anchor: &Anchor) -> ferx_core::types::Population {
    read_nonmem_csv(
        Path::new(&format!("nonmem_anchor/{}.csv", anchor.name)),
        None,
        Some("OCC"),
    )
    .expect("the anchor data loads")
}

#[test]
fn ipred_at_nonmem_ebes_matches_nonmem() {
    let mut failures = Vec::new();
    for anchor in &ANCHORS {
        let est = estimates(anchor.name);
        let pop = population(anchor);
        let phi = phi(anchor.name);
        let nm_ipred = table_ipred(anchor.name);
        assert_eq!(pop.subjects.len(), 24, "{}: subject count", anchor.name);
        // IPRED per leg: [walk, twin].
        let mut legs: Vec<Vec<Vec<f64>>> = Vec::new();
        for (leg, twin) in [("walk", false), ("twin", true)] {
            let model = model_at(anchor, &est, twin);
            let mut worst = 0.0_f64;
            let mut n_obs = 0usize;
            let mut preds = Vec::new();
            for s in &pop.subjects {
                assert!(
                    model.ode_spec.is_none() && model.absorption_ode_equivalent.is_some(),
                    "{} subject {}: a closed-form model with an ODE twin",
                    anchor.name,
                    s.id
                );
                let (eta, _) = phi[&s.id];
                // [η_CL, η_V]; per occasion g: [κ_CL = ETA(3+g), κ_ABS = ETA(6+g)].
                let kappas: Vec<Vec<f64>> = (0..3).map(|g| vec![eta[2 + g], eta[5 + g]]).collect();
                let got = ferx_core::pk::predict_iov(
                    &model,
                    s,
                    &model.default_params.theta,
                    &eta[..2],
                    &kappas,
                );
                let want = &nm_ipred[&s.id];
                assert_eq!(got.len(), want.len(), "{} subject {}", anchor.name, s.id);
                for (j, (&g, &w)) in got.iter().zip(want).enumerate() {
                    assert!(
                        g.is_finite() && w.is_finite() && w > 0.0,
                        "{} {leg} subject {} obs {j}: ferx {g}, NONMEM {w}",
                        anchor.name,
                        s.id
                    );
                    worst = worst.max((g - w).abs() / w);
                    n_obs += 1;
                }
                preds.push(got);
            }
            println!(
                "#1569/#1560 {} ({leg}): max rel |ferx IPRED − NONMEM IPRED| at NONMEM's EBEs = \
                 {worst:.3e} over {n_obs} observations",
                anchor.name
            );
            if !(worst < IPRED_TOL) {
                failures.push(format!(
                    "{} ({leg}): ferx departs from NONMEM's fixed-at-dose $DES by {worst:.3e}",
                    anchor.name
                ));
            }
            legs.push(preds);
        }
        // Two engines, so not one bit pattern: a walk leg that took the twin (or the reverse)
        // would make the two identical and this check vacuous for one of them.
        let differ = legs[0]
            .iter()
            .flatten()
            .zip(legs[1].iter().flatten())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(
            differ > 0,
            "{}: the walk and twin legs are bit-identical — one of them ran the other's engine",
            anchor.name
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

/// **Nightly** (`slow-tests`), unlike the IPRED check above. It is one evaluation, so Tier 2
/// by contract, but it measured **~205 s** in `Tests + coverage (core)` (run 36493929213:
/// 22:53:50 → 22:57:15, where the IPRED check in the same binary took 0.25 s), against 3.4 s
/// for both tests under a local `--profile ci-cov`. `AGENTS.md` allows the gate on that
/// measured cost plus a per-PR test that dies on the same mutation, and every mutation that
/// kills this test in the #1569 sweep kills one that runs on every PR. Re-reading the kernel
/// per segment in the value engine (M1), giving every dose dose 0's kernel (M7), or a
/// `PreparedForcings::get` that ignores the dose (M11) reddens
/// `ipred_at_nonmem_ebes_matches_nonmem`. A per-segment forcing on the `Dual2` walk (M2)
/// reddens 15 `sens::` unit tests, among them `iov_absorption_kappa_lands_on_the_dose_occasion`
/// and both `…_onset_on_a_covariate_record_returns_a_one_sided_derivative` fixtures.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: ~205 s in CI; opt in with --features slow-tests"
)]
fn objective_at_nonmem_optimum_matches_nonmem() {
    let mut failures = Vec::new();
    for anchor in &ANCHORS {
        let est = estimates(anchor.name);
        let model = model_at(anchor, &est, false);
        let pop = population(anchor);
        let phi = phi(anchor.name);
        let mut opts = FitOptions::default();
        opts.method = EstimationMethod::FoceI;
        opts.interaction = true; // NONMEM METHOD=1 INTER
        opts.run_covariance_step = false;
        opts.ode_reltol = 1e-12; // NONMEM ADVAN13 TOL=12
        opts.ode_abstol = 1e-12;
        opts.inner_tol = 1e-8;
        opts.verbose = false;
        let result = fit(&model, &pop, &model.default_params, &opts)
            .expect("the fixed-parameter objective evaluates");
        let gap = (result.ofv - est.ofv).abs();
        let mut worst_obj = 0.0_f64;
        for subj in &result.subjects {
            let (_, nm_obj) = phi[&subj.id];
            assert!(subj.ofv_contribution.is_finite(), "{}", anchor.name);
            worst_obj = worst_obj.max((subj.ofv_contribution - nm_obj).abs());
        }
        println!(
            "#1569 {}: ferx OFV {:.6} vs NONMEM {:.6} (|gap| {gap:.3e}); max |Δ per-subject OBJ| \
             {worst_obj:.3e}",
            anchor.name, result.ofv, est.ofv
        );
        if !(result.ofv.is_finite() && gap < OFV_TOL) {
            failures.push(format!(
                "{}: ferx FOCEI objective {:.6} vs NONMEM {:.6}",
                anchor.name, result.ofv, est.ofv
            ));
        }
        if !(worst_obj < OBJ_TOL) {
            failures.push(format!(
                "{}: a per-subject objective contribution departs from NONMEM's by \
                 {worst_obj:.3e}",
                anchor.name
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

// Bounds, from the measured agreement (macOS arm64, NONMEM `anchor` config):
//
// | anchor          | IPRED max rel (twin / walk leg) | OFV gap | max |Δ per-subject OBJ| |
// |-----------------|---------------------------------|---------|-------------------------|
// | transit_iov_mtt | 6.0e-12 / 7.6e-12               | 1.0e-6  | 9.0e-7                  |
// | ig_iov_mat      | 8.6e-12 / 5.4e-12               | 2.2e-6  | 7.5e-7                  |
//
// (The walk leg of `ig_iov_mat` serves 13 of the 24 subjects; the other 11 are in the
// flip-flop regime for some interval and go to the twin — pinned in
// `pk::absorption_walk::tests::nonmem_anchor_populations_route_to_the_walk_at_nonmem_ebes`.
// The OFV / OBJ columns were measured on the twin, #1569; the objective check now runs the
// plain model, i.e. the walk — re-measured by the slow-tests run.)
//
// Each bound carries ~10x over the worse measurement. On the pre-#1569 engine — every open
// dose on the current occasion's kernel — the same checks miss by orders of magnitude more
// (see the PR); both engines evaluate the same tolerances (`TOL=12`, `ode_reltol = 1e-12`).
const IPRED_TOL: f64 = 1e-10;
const OFV_TOL: f64 = 2e-5;
const OBJ_TOL: f64 = 1e-5;
