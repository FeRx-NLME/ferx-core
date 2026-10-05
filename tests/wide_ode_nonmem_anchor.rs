//! NONMEM anchor for an ODE model with more individual parameters than the static
//! sensitivity walk covers (#1661).
//!
//! Before #1661 any ODE model with more than 12 individual parameters ran its whole
//! fit on finite differences: the static superposition walk's cap sat on the
//! model-level analytic gate, so the event-driven walk — sized on `θ + η`, not on the
//! individual-parameter count — was never reached. This fixture is built to trip
//! that cap the way real models do, on a small `θ + η`.
//!
//! ## The kit (`nonmem_anchor/`)
//!
//! - **Model** (`wide_ode_myelo.ctl` / `wide_ode_myelo_fit.ferx`): sequential PK/PD.
//!   - 2-cpt PK from data columns (`V1I K10I K12I K21I`), as in a two-stage analysis.
//!   - A 4-transit myelosuppression chain with linear drug effect and feedback.
//!   - The cell compartments start at baseline through `AMT=1` doses scaled by
//!     `F3..F7 = BAS`.
//!   - That is 14 individual parameters on 4 θ + 3 η.
//! - **Design** (`make_wide_ode_myelo_template.py`, stdlib, seed 1661): 24 subjects,
//!   three 150 mg IV boluses two weeks apart, 12 cell counts each, none on a dose
//!   time. Later doses land on residual drug and a depressed count.
//! - **Data** (`wide_ode_myelo_sim.ctl`): DV comes from **NONMEM's own
//!   `$SIMULATION`**, not from ferx; `wide_ode_myelo_from_sim.py` turns its table into
//!   `wide_ode_myelo.csv`, which both engines fit.
//! - **Reference** (`results/wide_ode_myelo.{lst,ext,phi}`): NONMEM 7.5.1 FOCEI
//!   (`INTERACTION`, ADVAN13 `TOL=10`) from starts deliberately off the truth.
//!
//! ## Result
//!
//! | | ferx | NONMEM 7.5.1 |
//! |---|---|---|
//! | OFV       | 95.6853  | 95.685257 |
//! | TVBAS     | 4.685479 | 4.68554   |
//! | TVMTT     | 83.905347| 83.9048   |
//! | TVSLOPE   | 0.087647 | 0.0876455 |
//! | TVGAM     | 0.157643 | 0.157643  |
//! | ω²(BAS)   | 0.075262 | 0.0752494 |
//! | ω²(MTT)   | 0.064207 | 0.0642430 |
//! | ω²(SLOPE) | 0.128038 | 0.128017  |
//! | σ (SD)    | 0.145764 | 0.145762  |
//!
//! ferx fits this in 7.6 s on the analytic route; with `gradient = fd`, which is the
//! route every such model took before #1661, it takes 92 s.
//!
//! NONMEM stops with `ROUNDING ERRORS (ERROR=134)`, the usual outcome on a perfectly
//! specified simulated fit, at final gradients ≤ 0.21. Its `$COV MATRIX=R` step then
//! aborts (R matrix algorithmically non-positive-semidefinite), so no SEs are
//! anchored here.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const MODEL_FILE: &str = "nonmem_anchor/wide_ode_myelo_fit.ferx";
const DATA_FILE: &str = "nonmem_anchor/wide_ode_myelo.csv";
const PHI_FILE: &str = "nonmem_anchor/results/wide_ode_myelo.phi";

/// NONMEM 7.5.1 final estimates, `results/wide_ode_myelo.ext`'s `-1000000000` row.
const NM_OFV: f64 = 95.685257254298918;
const NM_THETA: [f64; 4] = [4.68554, 83.9048, 0.0876455, 0.157643];
const NM_OMEGA: [f64; 3] = [0.0752494, 0.0642430, 0.128017];
/// `SIGMA(1,1)` is a variance in both NONMEM and the ferx model file's default scale.
const NM_SIGMA_VAR: f64 = 0.0212464;
const THETA_NAMES: [&str; 4] = ["TVBAS", "TVMTT", "TVSLOPE", "TVGAM"];

/// The ferx twin with its `[parameters]` initial estimates replaced by NONMEM's final
/// ones. Each replaced line is asserted present, so an edit to the model file cannot
/// silently leave a parameter at its starting value.
fn model_at_nonmem_optimum() -> String {
    let src = std::fs::read_to_string(MODEL_FILE).expect("anchor model file");
    let swaps = [
        (
            "theta TVBAS(4, 0.1, 100)",
            format!("theta TVBAS({}, 0.1, 100)", NM_THETA[0]),
        ),
        (
            "theta TVMTT(80, 1, 1000)",
            format!("theta TVMTT({}, 1, 1000)", NM_THETA[1]),
        ),
        (
            "theta TVSLOPE(0.06, 0.0001, 10)",
            format!("theta TVSLOPE({}, 0.0001, 10)", NM_THETA[2]),
        ),
        (
            "theta TVGAM(0.25, 0.01, 2)",
            format!("theta TVGAM({}, 0.01, 2)", NM_THETA[3]),
        ),
        (
            "omega ETA_BAS   ~ 0.05",
            format!("omega ETA_BAS   ~ {}", NM_OMEGA[0]),
        ),
        (
            "omega ETA_MTT   ~ 0.05",
            format!("omega ETA_MTT   ~ {}", NM_OMEGA[1]),
        ),
        (
            "omega ETA_SLOPE ~ 0.05",
            format!("omega ETA_SLOPE ~ {}", NM_OMEGA[2]),
        ),
        (
            "sigma PROP_ERR ~ 0.04",
            format!("sigma PROP_ERR ~ {NM_SIGMA_VAR}"),
        ),
    ];
    let mut out = src;
    for (from, to) in swaps {
        assert_eq!(
            out.matches(from).count(),
            1,
            "anchor model line not found: {from}"
        );
        out = out.replace(from, &to);
    }
    out
}

/// NONMEM's POSTHOC EBEs at its final estimates, `(ID, [η₁, η₂, η₃])`, from `.phi`.
fn nonmem_ebes() -> Vec<(String, [f64; 3])> {
    let text = std::fs::read_to_string(PHI_FILE).expect("NONMEM .phi");
    text.lines()
        .skip(2)
        .map(|l| {
            let v: Vec<&str> = l.split_whitespace().collect();
            let eta = |i: usize| v[2 + i].parse::<f64>().expect("eta");
            (v[1].to_string(), [eta(0), eta(1), eta(2)])
        })
        .collect()
}

fn options(outer_maxiter: Option<usize>) -> FitOptions {
    let mut opts = FitOptions {
        method: EstimationMethod::FoceI,
        run_covariance_step: false,
        verbose: false,
        ..Default::default()
    };
    if let Some(n) = outer_maxiter {
        opts.outer_maxiter = n;
    }
    opts
}

/// At NONMEM's final estimates, ferx's FOCEI objective and every subject's EBE must
/// match NONMEM's. This is the inner loop measured against an external reference.
///
/// The finite-difference inner loop this model used to get is what left EBEs short
/// on the reported model (61 of 165 subjects at visibly worse individual
/// objectives). The test also pins that this fixture really runs on the analytic
/// route. A fit routed to FD would say nothing about the code #1661 changed.
#[test]
fn wide_ode_model_matches_nonmem_at_their_optimum() {
    let parsed = parse_full_model(&model_at_nonmem_optimum()).expect("anchor model must parse");
    let model = parsed.model;
    assert!(
        model.pk_indices.len() > 12,
        "the fixture must exceed the static walk's 12 individual parameters, has {}",
        model.pk_indices.len()
    );
    let pop = read_nonmem_csv(Path::new(DATA_FILE), None, None).expect("anchor data");

    let result = fit(&model, &pop, &model.default_params, &options(Some(0)))
        .expect("evaluation at NONMEM's optimum must run");

    assert_eq!(
        result.gradient_method_inner, "analytic (Dual2)",
        "#1661: a model past 12 individual parameters must take the analytic inner route"
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("finite-difference")),
        "no subject may fall back to FD: {:?}",
        result.warnings
    );

    // Measured: ferx 95.685276 against NONMEM 95.685257, a gap of 1.9e-5. The bound
    // keeps ~50× headroom over that, and stays far below a single subject's EBE being
    // left short: on the reported model the FD inner loop cost 1–118 per subject.
    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        (result.ofv - NM_OFV).abs() < 1e-3,
        "ferx OFV {:.6} at NONMEM's optimum vs NONMEM {NM_OFV:.6}",
        result.ofv
    );

    // Measured worst gap over 24 subjects × 3 etas: 1.7e-6. Bound 1e-4 (~60×).
    let nm = nonmem_ebes();
    assert_eq!(nm.len(), result.subjects.len());
    for ((id, nm_eta), s) in nm.iter().zip(&result.subjects) {
        assert_eq!(*id, s.id, "subject order must match NONMEM's");
        for k in 0..3 {
            let d = (s.eta[k] - nm_eta[k]).abs();
            assert!(
                d.is_finite() && d < 1e-4,
                "subject {id} eta {k}: ferx {} vs NONMEM {}",
                s.eta[k],
                nm_eta[k]
            );
        }
    }
}

fn assert_rel(name: &str, got: f64, want: f64, tol: f64) {
    let rel = (got - want).abs() / want.abs();
    assert!(
        rel.is_finite() && rel < tol,
        "{name}: ferx {got:.6} vs NONMEM {want:.6} (relative gap {rel:.2e}, bound {tol:.0e})"
    );
}

/// A free fit from the same starts must land on NONMEM's optimum.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: full FOCEI convergence fit (#1661); opt in with --features slow-tests"
)]
fn wide_ode_model_fit_matches_nonmem() {
    let model = parse_full_model(&std::fs::read_to_string(MODEL_FILE).expect("model"))
        .expect("anchor model must parse")
        .model;
    let pop = read_nonmem_csv(Path::new(DATA_FILE), None, None).expect("anchor data");
    let result = fit(&model, &pop, &model.default_params, &options(None)).expect("fit");

    assert_eq!(result.gradient_method_inner, "analytic (Dual2)");

    // Measured gaps, ferx against NONMEM:
    //   OFV 95.685274 vs 95.685257 (1.7e-5);
    //   θ: largest relative gap 1.3e-5 (TVBAS);
    //   ω²: largest 5.6e-4 (MTT; 0.064207 vs 0.064243);
    //   σ: 2e-5.
    // NONMEM itself stops at ERROR=134 with gradients up to 0.21, so the ω gap is the
    // flatness of the optimum, not a difference in the objective. Bounds keep ≥ 9×
    // headroom over each measured gap.
    assert!(
        result.ofv.is_finite() && (result.ofv - NM_OFV).abs() < 1e-3,
        "ferx OFV {:.6} vs NONMEM {NM_OFV:.6}",
        result.ofv
    );
    for (i, name) in THETA_NAMES.iter().enumerate() {
        assert_rel(name, result.theta[i], NM_THETA[i], 1e-3);
    }
    for (k, name) in ["omega BAS", "omega MTT", "omega SLOPE"].iter().enumerate() {
        assert_rel(name, result.omega[(k, k)], NM_OMEGA[k], 5e-3);
    }
    assert_rel("sigma (SD)", result.sigma[0], NM_SIGMA_VAR.sqrt(), 1e-3);
}
