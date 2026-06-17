//! NONMEM cross-check for the **compartment-state vector** exposed in
//! `[derived]` (issue #205).
//!
//! The unit/integration tests in `tests/compartment_states.rs` assert
//! *internal* invariants (peripheral ≥ 0, `compartments[central] == IPRED`,
//! named access == subscript access). Those are necessary but not sufficient:
//! they do not pin the **absolute value** of the peripheral / depot states, so
//! a wrong residue coefficient in `two_cpt_oral_peripheral` (review finding #1
//! on PR #207) would pass them. This file closes that gap by comparing ferx's
//! analytical peripheral/depot states against NONMEM 7.5.1.
//!
//! ## How the reference was produced
//!
//! NONMEM evaluates the model at the fixed θ with η = 0 (`$OMEGA 0 FIX`,
//! `$ESTIMATION MAXEVAL=0`) and tables the raw compartment **amounts** `A(i)`.
//! ferx exposes central/peripheral compartments as **concentrations**
//! (`A(i)/V`) and the depot as an amount (see `docs/src/model-file/derived.md`),
//! so the check rescales ferx's `compartments[i]` by the matching volume before
//! comparing to `A(i)`. ferx is likewise evaluated at η = 0 (fixed θ via
//! `(.. , FIX)`, `omega ~ 0.0`).
//!
//! The four control files below were run with NONMEM 7.5.1 (`nmfe75`); the
//! `A(i)` columns from each `*.tab` are transcribed verbatim into the
//! `*_A_*` arrays. Re-running NONMEM with these control files reproduces the
//! numbers to all printed digits. The `.ctl` + `.csv` pairs are committed under
//! `tests/nonmem/` (`iv2`, `iv3`, `oral2`, `oral3`) for regeneration.
//!
//! ### `iv2.ctl` — 2-cpt IV bolus (ADVAN3 TRANS4); CL=2 V1=10 Q=1 V2=20
//! ```text
//! $SUBROUTINES ADVAN3 TRANS4
//! $PK
//!   CL=THETA(1)  V1=THETA(2)  Q=THETA(3)  V2=THETA(4)  S1=V1
//! $ERROR
//!   A1=A(1)  A2=A(2)  IPRED=F  Y=IPRED+EPS(1)
//! $THETA 2.0 FIX 10.0 FIX 1.0 FIX 20.0 FIX
//! $OMEGA 0 FIX   $SIGMA 1 FIX
//! $ESTIMATION MAXEVAL=0 METHOD=0 NOABORT
//! ```
//! Dose `AMT=100, CMT=1` at t=0; `A(1)`=central amount, `A(2)`=peripheral amount.
//!
//! ### `iv3.ctl` — 3-cpt IV bolus (ADVAN11 TRANS4); CL=5 V1=10 Q2=2 V2=20 Q3=1.5 V3=30
//! Dose `AMT=100, CMT=1` (central); `A(1)`=central, `A(2)`=periph1, `A(3)`=periph2.
//! ferx params: `cl=5, v1=10, q2=2, v2=20, q3=1.5, v3=30`.
//!
//! ### `oral2.ctl` — 2-cpt oral (ADVAN4 TRANS4); CL=2 V2=10 Q=1 V3=20 KA=1
//! ```text
//! $SUBROUTINES ADVAN4 TRANS4
//! $PK CL=THETA(1) V2=THETA(2) Q=THETA(3) V3=THETA(4) KA=THETA(5) S2=V2
//! ```
//! Dose `AMT=100, CMT=1` (depot); `A(1)`=depot, `A(2)`=central, `A(3)`=peripheral.
//! ferx params: `cl=2, v1=10 (=NM V2), q=1, v2=20 (=NM V3), ka=1`.
//!
//! ### `oral3.ctl` — 3-cpt oral (ADVAN12 TRANS4); CL=5 V2=10 Q3=2 V3=20 Q4=1.5 V4=30 KA=1
//! ```text
//! $SUBROUTINES ADVAN12 TRANS4
//! $PK CL=THETA(1) V2=THETA(2) Q3=THETA(3) V3=THETA(4) Q4=THETA(5) V4=THETA(6) KA=THETA(7) S2=V2
//! ```
//! Dose `AMT=100, CMT=1` (depot); `A(1)`=depot, `A(2)`=central, `A(3)`=periph1,
//! `A(4)`=periph2. ferx params: `cl=5, v1=10, q2=2, v2=20, q3=1.5, v3=30, ka=1`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, Population};
use ferx_core::{fit, FitOptions};

mod common;

/// Combined absolute+relative tolerance. The depot amount decays to ~4e-9 by
/// t=24, where a pure relative tolerance is meaningless, so an absolute floor
/// is added. ferx (closed-form Bateman) and NONMEM (ADVAN ODE integration)
/// agree far inside this band for the non-tiny states.
fn close(ferx: f64, nm: f64) -> bool {
    (ferx - nm).abs() <= 1e-4 * nm.abs() + 1e-6
}

/// Build a one-subject population. `observations` are set to NONMEM's PRED at
/// η = 0 so that the EBE inner optimizer's exact minimizer is η = 0 (residuals
/// and the prior penalty gradient both vanish there) — this pins ferx to the
/// same η = 0 point NONMEM evaluated under `$OMEGA 0 FIX`, independent of the
/// (tiny) omega and the single outer iteration.
fn single_subject_pop(obs_times: Vec<f64>, observations: Vec<f64>, dose_cmt: usize) -> Population {
    let n = obs_times.len();
    assert_eq!(obs_times.len(), observations.len());
    Population {
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![common::subject(
            "1",
            vec![DoseEvent::new(0.0, 100.0, dose_cmt, 0.0, false, 0.0)],
            obs_times,
            observations,
            vec![1; n],
        )],
    }
}

/// Fit at fixed θ / η = 0 and return the named derived column for subject 1.
fn derived_col(model_src: &str, pop: &Population, name: &str) -> Vec<f64> {
    let model = parse_model_string(model_src).expect("model must parse");
    let mut opts = FitOptions::default();
    opts.verbose = false;
    let result = fit(&model, pop, &model.default_params, &opts).expect("fit must not error");
    let sr = &result.subjects[0];
    sr.extra_columns
        .iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("derived column `{name}` must exist"))
        .1
        .clone()
}

// ── NONMEM reference amounts (transcribed from *.tab) ─────────────────────────

const IV_TIMES: [f64; 4] = [1.0, 4.0, 12.0, 24.0];
// 2-cpt IV: A(1) central amount, A(2) peripheral amount.
const IV_A_CENTRAL: [f64; 4] = [74.283560663, 31.862421526, 6.4905561668, 3.0959062573];
const IV_A_PERIPH: [f64; 4] = [8.4234563155, 20.974068151, 23.128480554, 16.375499360];

// 3-cpt IV (`iv3.ctl`, ADVAN11 TRANS4; CL=5 V1=10 Q2=2 V2=20 Q3=1.5 V3=30):
// A(1) central, A(2) periph1, A(3) periph2.
const IV3_A_CENTRAL: [f64; 4] = [43.514217307, 6.1538551883, 2.2707729196, 1.3153095914];
const IV3_A_PERIPH1: [f64; 4] = [12.789371770, 18.120069615, 11.437952547, 5.7515008038];
const IV3_A_PERIPH2: [f64; 4] = [9.8656916985, 15.633077201, 13.523442982, 9.7158855197];

const ORAL_TIMES: [f64; 7] = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
// 2-cpt oral: A(1) depot, A(2) central, A(3) peripheral.
const ORAL2_A_DEPOT: [f64; 7] = [
    60.653065971,
    36.787944117,
    13.533528324,
    1.8315638889,
    0.033546262790,
    0.00061442123533,
    3.7751345443e-09,
];
const ORAL2_A_CENTRAL: [f64; 7] = [
    36.319547282,
    53.332822580,
    59.374568299,
    41.637289920,
    15.884952915,
    7.5894251978,
    3.2156605649,
];
const ORAL2_A_PERIPH: [f64; 7] = [
    1.0032394980,
    3.2528147973,
    8.7904159387,
    17.695093851,
    23.973668515,
    23.546882142,
    16.898877162,
];

// 3-cpt oral: A(1) depot, A(2) central, A(3) periph1, A(4) periph2.
const ORAL3_A_DEPOT: [f64; 7] = ORAL2_A_DEPOT; // identical dosing/KA → identical depot
const ORAL3_A_CENTRAL: [f64; 7] = [
    31.533338873,
    39.922118232,
    32.533946284,
    12.396721559,
    3.3199561377,
    2.3982934860,
    1.3754367745,
];
const ORAL3_A_PERIPH1: [f64; 7] = [
    1.8163070319,
    5.3388312327,
    11.953778247,
    17.378583631,
    15.482735194,
    12.175881952,
    6.0849038046,
];
const ORAL3_A_PERIPH2: [f64; 7] = [
    1.3744591951,
    4.0814462953,
    9.3630539737,
    14.498493707,
    15.104288402,
    13.856525401,
    10.010073786,
];

fn assert_matches(label: &str, ferx_amounts: &[f64], nm: &[f64]) {
    assert_eq!(
        ferx_amounts.len(),
        nm.len(),
        "{label}: length {} != NONMEM length {}",
        ferx_amounts.len(),
        nm.len()
    );
    for (j, (&f, &n)) in ferx_amounts.iter().zip(nm.iter()).enumerate() {
        assert!(
            close(f, n),
            "{label}: obs {j}: ferx {f:.10} vs NONMEM {n:.10} \
             (abs diff {:.3e}, rel {:.3e})",
            (f - n).abs(),
            (f - n).abs() / n.abs().max(1e-30),
        );
    }
}

/// 2-cpt IV: central concentration × V1 and peripheral concentration × V2 must
/// reproduce NONMEM `A(1)` and `A(2)`.
#[test]
fn nonmem_2cpt_iv_compartment_amounts() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, FIX)
  theta V1(10.0, FIX)
  theta Q(1.0, FIX)
  theta V2(20.0, FIX)
  omega ETA_CL ~ 0.0
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V1 = V1
  Q  = Q
  V2 = V2

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ proportional(PROP)

[derived]
  C_central = compartments[0]
  C_periph  = compartments[1]

[fit_options]
  method   = focei
  maxiter  = 1
  gradient = fd
";
    let v1 = 10.0;
    let v2 = 20.0;
    let obs = IV_A_CENTRAL.iter().map(|a| a / v1).collect();
    let pop = single_subject_pop(IV_TIMES.to_vec(), obs, 1);
    let central_amt: Vec<f64> = derived_col(MODEL, &pop, "C_central")
        .iter()
        .map(|c| c * v1)
        .collect();
    let periph_amt: Vec<f64> = derived_col(MODEL, &pop, "C_periph")
        .iter()
        .map(|c| c * v2)
        .collect();
    assert_matches("2cpt-iv central A(1)", &central_amt, &IV_A_CENTRAL);
    assert_matches("2cpt-iv peripheral A(2)", &periph_amt, &IV_A_PERIPH);
}

/// 3-cpt IV: central × V1, periph1 × V2, periph2 × V3 must reproduce NONMEM
/// `A(1)`..`A(3)`. Validates `three_cpt_iv_peripherals`.
#[test]
fn nonmem_3cpt_iv_compartment_amounts() {
    const MODEL: &str = "
[parameters]
  theta CL(5.0, FIX)
  theta V1(10.0, FIX)
  theta Q2(2.0, FIX)
  theta V2(20.0, FIX)
  theta Q3(1.5, FIX)
  theta V3(30.0, FIX)
  omega ETA_CL ~ 0.0
  sigma PROP   ~ 0.04 (sd)

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V1 = V1
  Q2 = Q2
  V2 = V2
  Q3 = Q3
  V3 = V3

[structural_model]
  pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3)

[error_model]
  DV ~ proportional(PROP)

[derived]
  C_central = compartments[0]
  C_periph1 = compartments[1]
  C_periph2 = compartments[2]

[fit_options]
  method   = focei
  maxiter  = 1
  gradient = fd
";
    let (v1, v2, v3) = (10.0, 20.0, 30.0);
    let obs = IV3_A_CENTRAL.iter().map(|a| a / v1).collect();
    let pop = single_subject_pop(IV_TIMES.to_vec(), obs, 1);
    let central_amt: Vec<f64> = derived_col(MODEL, &pop, "C_central")
        .iter()
        .map(|c| c * v1)
        .collect();
    let periph1_amt: Vec<f64> = derived_col(MODEL, &pop, "C_periph1")
        .iter()
        .map(|c| c * v2)
        .collect();
    let periph2_amt: Vec<f64> = derived_col(MODEL, &pop, "C_periph2")
        .iter()
        .map(|c| c * v3)
        .collect();
    assert_matches("3cpt-iv central A(1)", &central_amt, &IV3_A_CENTRAL);
    assert_matches("3cpt-iv periph1 A(2)", &periph1_amt, &IV3_A_PERIPH1);
    assert_matches("3cpt-iv periph2 A(3)", &periph2_amt, &IV3_A_PERIPH2);
}

/// 2-cpt oral: depot amount, central conc × V1, peripheral conc × V2 must
/// reproduce NONMEM `A(1)`, `A(2)`, `A(3)`. This is the **critical** check for
/// review finding #1 (`two_cpt_oral_peripheral` formula correctness).
#[test]
fn nonmem_2cpt_oral_compartment_amounts() {
    const MODEL: &str = "
[parameters]
  theta CL(2.0, FIX)
  theta V1(10.0, FIX)
  theta Q(1.0, FIX)
  theta V2(20.0, FIX)
  theta KA(1.0, FIX)
  omega ETA_CL ~ 0.0
  sigma PROP   ~ 0.01

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V1 = V1
  Q  = Q
  V2 = V2
  KA = KA

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[derived]
  A_depot   = compartments[0]
  C_central = compartments[1]
  C_periph  = compartments[2]

[fit_options]
  method   = focei
  maxiter  = 1
  gradient = fd
";
    let v1 = 10.0;
    let v2 = 20.0;
    let obs = ORAL2_A_CENTRAL.iter().map(|a| a / v1).collect();
    let pop = single_subject_pop(ORAL_TIMES.to_vec(), obs, 1);
    let depot_amt = derived_col(MODEL, &pop, "A_depot");
    let central_amt: Vec<f64> = derived_col(MODEL, &pop, "C_central")
        .iter()
        .map(|c| c * v1)
        .collect();
    let periph_amt: Vec<f64> = derived_col(MODEL, &pop, "C_periph")
        .iter()
        .map(|c| c * v2)
        .collect();
    assert_matches("2cpt-oral depot A(1)", &depot_amt, &ORAL2_A_DEPOT);
    assert_matches("2cpt-oral central A(2)", &central_amt, &ORAL2_A_CENTRAL);
    assert_matches("2cpt-oral peripheral A(3)", &periph_amt, &ORAL2_A_PERIPH);
}

/// 3-cpt oral: depot, central × V1, periph1 × V2, periph2 × V3 must reproduce
/// NONMEM `A(1)`..`A(4)`. Validates `three_cpt_oral_peripherals`.
#[test]
fn nonmem_3cpt_oral_compartment_amounts() {
    const MODEL: &str = "
[parameters]
  theta CL(5.0, FIX)
  theta V1(10.0, FIX)
  theta Q2(2.0, FIX)
  theta V2(20.0, FIX)
  theta Q3(1.5, FIX)
  theta V3(30.0, FIX)
  theta KA(1.0, FIX)
  omega ETA_CL ~ 0.0
  sigma PROP   ~ 0.04 (sd)

[individual_parameters]
  CL = CL * exp(ETA_CL)
  V1 = V1
  Q2 = Q2
  V2 = V2
  Q3 = Q3
  V3 = V3
  KA = KA

[structural_model]
  pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA)

[error_model]
  DV ~ proportional(PROP)

[derived]
  A_depot   = compartments[0]
  C_central = compartments[1]
  C_periph1 = compartments[2]
  C_periph2 = compartments[3]

[fit_options]
  method   = focei
  maxiter  = 1
  gradient = fd
";
    let (v1, v2, v3) = (10.0, 20.0, 30.0);
    let obs = ORAL3_A_CENTRAL.iter().map(|a| a / v1).collect();
    let pop = single_subject_pop(ORAL_TIMES.to_vec(), obs, 1);
    let depot_amt = derived_col(MODEL, &pop, "A_depot");
    let central_amt: Vec<f64> = derived_col(MODEL, &pop, "C_central")
        .iter()
        .map(|c| c * v1)
        .collect();
    let periph1_amt: Vec<f64> = derived_col(MODEL, &pop, "C_periph1")
        .iter()
        .map(|c| c * v2)
        .collect();
    let periph2_amt: Vec<f64> = derived_col(MODEL, &pop, "C_periph2")
        .iter()
        .map(|c| c * v3)
        .collect();
    assert_matches("3cpt-oral depot A(1)", &depot_amt, &ORAL3_A_DEPOT);
    assert_matches("3cpt-oral central A(2)", &central_amt, &ORAL3_A_CENTRAL);
    assert_matches("3cpt-oral periph1 A(3)", &periph1_amt, &ORAL3_A_PERIPH1);
    assert_matches("3cpt-oral periph2 A(4)", &periph2_amt, &ORAL3_A_PERIPH2);
}
