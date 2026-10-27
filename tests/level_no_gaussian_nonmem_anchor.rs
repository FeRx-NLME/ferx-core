//! NONMEM anchor for a `theta NAME[COL]` level block read by subjects with **no Gaussian
//! observation** (#1797).
//!
//! Reference: `nonmem_anchor/level_no_gaussian.ctl`, NONMEM 7.6.0 `ADVAN13 TOL=9`,
//! `$OMEGA 0 FIX`, `MAXEVAL=0 POSTHOC`, on `nonmem_anchor/level_no_gaussian.csv`: studies
//! 1–3 with two joint PK-TTE subjects each, plus TTE-only subjects 98 (`STUDY=1`) and 99
//! (`STUDY=3`), one exact event each at t = 7. `#OBJV = 50.593`. Outputs in
//! `nonmem_anchor/results/`.
//!
//! NONMEM has no level-index object: `$PK` reads the record's own `STUDY`, so its value is
//! the closed form the fix restores. Before #1797 ferx indexed 98 **and** 99 at level 1, so
//! 99's hazard read `STUDY=1`'s θ: its objective was 6.691465 (98's) against NONMEM's
//! 5.894262, a 0.797202 gap this anchor fails on. 98 is the control — level 1 either way.
//!
//! Compared object: per-subject `OBJ` from the `.phi` against `2 × individual_nll`, which is
//! comparable with no constant (measured to be zero on `pktte_tdep`, #1166). The joint
//! subjects 1–6 are compared too, so a fix that broke a Gaussian subject's index would fail
//! here as well. Engine: the ferx side is the value path (`individual_nll` → the hazard's
//! ODE solve), through `bind_theta_levels`; no sensitivity is involved.

#![cfg(feature = "survival")]

use ferx_core::api::read_population_for;
use ferx_core::bind_theta_levels;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::stats::likelihood::individual_nll;

const MODEL: &str = "nonmem_anchor/level_no_gaussian_fit.ferx";
const DATA: &str = "nonmem_anchor/level_no_gaussian.csv";
const PHI: &str = "nonmem_anchor/results/level_no_gaussian.phi";

/// `(ID, OBJ)` per subject from the `.phi`.
fn nonmem_obj() -> Vec<(String, f64)> {
    std::fs::read_to_string(PHI)
        .expect("NONMEM .phi")
        .lines()
        .skip(2)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (f[1].to_string(), f[4].parse().expect("OBJ is numeric"))
        })
        .collect()
}

#[test]
fn per_subject_objective_matches_nonmem_for_tte_only_subjects() {
    let text = std::fs::read_to_string(MODEL).expect("anchor model file");
    let mut parsed = parse_full_model(&text).expect("anchor model parses");
    let (mut pop, _) = read_population_for(&parsed.model, &None, DATA, None, None, None, &[])
        .expect("anchor data reads");
    bind_theta_levels(&mut parsed, &text, &mut pop).expect("the block binds");
    let m = &parsed.model;
    assert!(
        m.eta_names.is_empty(),
        "evaluated at eta = 0, as NONMEM's `$OMEGA 0 FIX`"
    );

    let mut p = m.default_params.clone();
    for (name, v) in [("PLACEBO[STUDY=2]", 0.5), ("PLACEBO[STUDY=3]", 1.0)] {
        let k = m.theta_names.iter().position(|n| n == name).expect(name);
        p.theta[k] = v;
    }

    let obj = nonmem_obj();
    assert_eq!(obj.len(), 8, "eight subjects in the reference run");
    for id in ["98", "99"] {
        let s = pop.subjects.iter().find(|s| s.id == id).unwrap();
        assert!(
            s.obs_times.is_empty(),
            "subject {id} has no Gaussian observation"
        );
    }

    let (mut worst_joint, mut worst_tte_only) = (0.0f64, 0.0f64);
    for (id, want) in &obj {
        let s = pop.subjects.iter().find(|s| &s.id == id).unwrap();
        let got = 2.0 * individual_nll(m, s, &p.theta, &[], &p.omega, &p.sigma.values);
        // Checked before folding: `f64::max` drops a NaN, which would pass every bound.
        assert!(got.is_finite(), "subject {id}: 2·nll = {got}");
        let err = (got - want).abs();
        eprintln!("subject {id}: ferx {got:.12}  NONMEM {want:.12}  |Δ| {err:.3e}");
        if s.obs_times.is_empty() {
            worst_tte_only = worst_tte_only.max(err);
        } else {
            worst_joint = worst_joint.max(err);
        }
    }
    // Measured on macOS arm64 at `d43afca9` + this fix (CI re-runs it on Linux): the
    // TTE-only pair agrees to 0 in every printed digit — a constant hazard has a closed
    // form on both sides — and the joint subjects to 9.32e-8, the ODE tolerance on the PK
    // records. Bounds ~10× above those (the TTE-only one at a few ULP of the objective).
    // A regression to level 1 on subject 99 is a 0.797202 gap.
    assert!(
        worst_tte_only < 1e-12,
        "TTE-only subjects vs NONMEM: worst {worst_tte_only:.3e} (measured 0)"
    );
    assert!(
        worst_joint < 1e-6,
        "joint subjects vs NONMEM: worst {worst_joint:.3e} (measured 9.32e-8)"
    );
}
