//! NONMEM anchor for clock-forward RTTE with a drug-driven ODE-accumulated hazard (#3152).
//!
//! Reference: `nonmem_anchor/pkrtte_fwd.ctl`, NONMEM 7.5.1 `ADVAN13 TOL=9`, `$OMEGA 0 FIX`,
//! `LAPLACE LIKELIHOOD MAXEVAL=0 POSTHOC`, on `nonmem_anchor/pkrtte_fwd.csv` (8 subjects, three
//! oral doses each, 0 to 5 events, censored at 36), `#OBJV = 155.187`. Outputs in
//! `nonmem_anchor/results/`.
//!
//! The per-subject `OBJ` from the `.phi` is `2 × individual_nll` with no constant, as on
//! `pktte_tdep` (#1166). The hazard reads the concentration, and the `BETA = 0` twin moves
//! every subject's objective far past the bound, so the PK link is live in the comparison.

#![cfg(feature = "survival")]

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::stats::likelihood::individual_nll;

const PHI: &str = "nonmem_anchor/results/pkrtte_fwd.phi";

/// `2 × individual_nll` per subject of the anchor model, with `edit` applied to its source.
fn ferx_obj(edit: impl Fn(String) -> String) -> Vec<f64> {
    let src = edit(std::fs::read_to_string("nonmem_anchor/pkrtte_fwd_fit.ferx").unwrap());
    let m = parse_model_string(&src).expect("anchor model parses");
    let (pop, _) =
        read_population_for(&m, &None, "nonmem_anchor/pkrtte_fwd.csv", None, None, None, &[])
            .expect("anchor data loads");
    let p = &m.default_params;
    pop.subjects
        .iter()
        .map(|s| 2.0 * individual_nll(&m, s, &p.theta, &[], &p.omega, &p.sigma.values))
        .collect()
}

#[test]
fn per_subject_objective_matches_nonmem() {
    let nonmem: Vec<f64> = std::fs::read_to_string(PHI)
        .expect("NONMEM .phi")
        .lines()
        .skip(2)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().last().expect("OBJ column").parse().expect("numeric"))
        .collect();
    let live = ferx_obj(|s| s);
    let flat = ferx_obj(|s| s.replace("TVBETA(0.2, FIX)", "TVBETA(0.0, FIX)"));
    assert_eq!(live.len(), nonmem.len(), "one OBJ per subject");
    let (mut worst, mut gap) = (0.0f64, f64::INFINITY);
    for ((f, z), o) in live.iter().zip(&flat).zip(&nonmem) {
        // `f64::max` / `min` drop a NaN, so check finiteness first.
        assert!(f.is_finite() && z.is_finite() && o.is_finite(), "not finite: {f}, {z}, {o}");
        worst = worst.max((f - o).abs());
        gap = gap.min((z - o).abs());
    }
    eprintln!("per-subject |2·NLL − OBJ| worst {worst:.3e}; BETA = 0 twin nearest {gap:.3e}");
    assert!(worst < BOUND, "worst {worst:.3e}");
    assert!(gap > 1e3 * BOUND, "the PK link is live: {gap:.3e}");
}

// Measured (Linux x86_64): worst 1.10e-8, at the model file's ODE tolerances; the BETA = 0
// twin's nearest subject is 0.85 away. About 18×.
const BOUND: f64 = 2e-7;
