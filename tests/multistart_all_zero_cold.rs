//! #2014 — an estimating fit runs the guarded inner multi-start on its first evaluation.
//!
//! The outer loop hands the inner solve `Some(zeros)` on its first evaluation, and the
//! multi-start was gated on `eta_init.is_none()`, so before #2014 it never ran until the
//! final cold re-solve. On a subject with two EBE modes the optimizer then minimised the
//! population objective with that subject in the shallower one, the final re-solve found the
//! deeper one, and the #1565 basin restart re-entered the shallower mode from a zero start
//! again: the fit reported `converged = false` with `W_EBE_START_DEPENDENT`.
//!
//! The fixture is three subjects of a one-compartment oral model. Subject 1 is flip-flop
//! (ka = 0.02 < k = CL/V = 0.1), so swapping ka and k (V' = V·k/ka) reproduces its curve
//! exactly and its individual objective has two modes that differ only in the Ω penalty.
//! Subjects 2 and 3 are ordinary controls. Each subject has an inert `EVID=3` reset before
//! its dose, which puts it on the multi-start's scan-every-coordinate arm: without it the
//! well-curved twin is never seeded at all (#2013, a separate defect). Only `TVCL` is
//! estimated. NONMEM 7.5.1 at the same estimates, `MAXEVAL=0`, gives −219.629822 from
//! η = 0 (`MCETA=0`) and −235.253006 with `MCETA=1000`; ferx matches both to 6 decimals at
//! `maxiter = 0` (#2013's comparison).
//!
//! Measured on macOS arm64, this test in a debug build: with the gate reverted to
//! `eta_init.is_none()` the fit reports OFV −235.559329, `converged = false`,
//! `W_EBE_START_DEPENDENT` and `TVCL = 1.065939` (the optimum of the trapped objective),
//! after 17 outer iterations; with the fix, −235.559336, `converged = true` and
//! `TVCL = 1.066269`, after 11. Both run in 0.4 s.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, FitResult};

const TIMES: [f64; 15] = [
    0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 48.0, 72.0, 96.0,
];

fn conc(cl: f64, v: f64, ka: f64, t: f64) -> f64 {
    let k = cl / v;
    100.0 * ka / (v * (ka - k)) * ((-k * t).exp() - (-ka * t).exp())
}

fn run(inner_restarts: usize) -> FitResult {
    let mut csv = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV\n");
    for (id, (cl, v, ka)) in [(1.0, 10.0, 0.02), (1.0, 10.0, 1.0), (1.2, 9.0, 0.8)]
        .into_iter()
        .enumerate()
    {
        let id = id + 1;
        csv.push_str(&format!("{id},0,.,3,.,1,1\n{id},0,.,1,100,1,1\n"));
        for t in TIMES {
            csv.push_str(&format!("{id},{t},{:.10e},0,.,1,0\n", conc(cl, v, ka, t)));
        }
    }
    let path = std::env::temp_dir().join(format!(
        "ferx_2014_multistart_{}_{inner_restarts}.csv",
        std::process::id()
    ));
    std::fs::write(&path, csv).expect("write the fixture data");
    let pop = read_nonmem_csv(&path, None, None).expect("the fixture data loads");
    let _ = std::fs::remove_file(&path);
    assert!(
        pop.subjects.iter().all(|s| s.has_resets()),
        "premise: every subject takes the scan-every-coordinate arm"
    );

    let parsed = parse_full_model(&format!(
        "[parameters]
  theta TVCL(1.0, 0.01, 100)
  theta TVV(10.0, 0.1, 1000) FIX
  theta TVKA(1.0, 0.001, 100) FIX
  omega ETA_CL ~ 0.04 FIX
  omega ETA_V  ~ 0.1 FIX
  omega ETA_KA ~ 1.0 FIX
  sigma PROP_ERR ~ 0.05 (sd) FIX
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method         = focei
  maxiter        = 200
  covariance     = false
  inner_restarts = {inner_restarts}
"
    ))
    .expect("the fixture model parses");
    let mut opts = parsed.fit_options;
    opts.verbose = false;
    fit(&parsed.model, &pop, &parsed.model.default_params, &opts).expect("the fit runs")
}

fn eta_ka(result: &FitResult, id: &str) -> f64 {
    let s = result
        .subjects
        .iter()
        .find(|s| s.id == id)
        .expect("subject present");
    s.eta[2]
}

#[test]
fn estimating_fit_multistarts_on_its_first_evaluation() {
    let fixed = run(1);
    let start_dependent: Vec<&String> = fixed
        .warnings
        .iter()
        .filter(|w| w.contains("W_EBE_START_DEPENDENT"))
        .collect();
    println!(
        "#2014 inner_restarts=1: OFV {:.9}, converged {}, {} iterations, TVCL {:.9}, \
         subject 1 η_KA {:.4}",
        fixed.ofv,
        fixed.converged,
        fixed.n_iterations,
        fixed.theta[0],
        eta_ka(&fixed, "1")
    );
    assert!(
        start_dependent.is_empty(),
        "the optimizer minimised under EBEs a cold re-solve beats (#2014): {start_dependent:?}"
    );
    assert!(
        fixed.converged,
        "the fit was restarted out of a trapped basin (#1565) instead of starting in the \
         deeper one (#2014)"
    );
    assert!(
        eta_ka(&fixed, "1") < -3.5,
        "subject 1 is not in the deeper mode: η_KA {}",
        eta_ka(&fixed, "1")
    );
    // Both references are this test's own run (macOS arm64). The OFV bound is a sanity
    // check only: the reported OFV is a cold re-solve either way, and the trapped fit's is
    // 7e-6 away. The TVCL bound is what separates the two optima: 1.065939 is 3.3e-4 away,
    // 3.3× the bound.
    assert!(
        (fixed.ofv - -235.559336).abs() < 1e-4,
        "OFV {:.6} vs −235.559336",
        fixed.ofv
    );
    assert!(
        (fixed.theta[0] - 1.066269).abs() < 1e-4,
        "TVCL {:.6} vs 1.066269 (1.065939 is the optimum of the trapped objective)",
        fixed.theta[0]
    );

    // The straddle: without the multi-start the same fit stays in the twin, so the
    // assertions above measure the multi-start and not the fixture.
    let trapped = run(0);
    println!(
        "#2014 inner_restarts=0: OFV {:.9}, converged {}, subject 1 η_KA {:.4}",
        trapped.ofv,
        trapped.converged,
        eta_ka(&trapped, "1")
    );
    assert!(
        eta_ka(&trapped, "1") > -3.0 && trapped.ofv > fixed.ofv + 10.0,
        "premise: inner_restarts = 0 must leave subject 1 in the twin (η_KA {}, OFV {:.6})",
        eta_ka(&trapped, "1"),
        trapped.ofv
    );
}
