//! NONMEM cross-check for an `[odes]` RHS that reads the `TAD` built-in while the
//! model carries an **estimated lagtime** — the cell #1070 is about.
//!
//! ## What is being anchored
//!
//! `eval_rhs_anchored` computes `tad = t − last_dose_eff` in `f64` and `eval_rhs_g`
//! writes it into the RHS variable table as a constant. But `last_dose_eff` **is**
//! the dose's lagged arrival (`d.time + lag`), so `∂TAD/∂lag = −1` never enters the
//! dual chain: the analytic η/θ gradient on the lag axis is wrong everywhere the
//! trajectory is integrated, not merely at an event boundary. The *value* stays
//! exact, which is why no prediction test ever caught it.
//!
//! Since FOCEI builds its `h` matrix from that same analytic `∂f/∂η`
//! (`inner_optimizer.rs`), the error lands in the **reported OFV** — so it is
//! NONMEM-anchorable, and that is what these tests do. #1070's interim fix routes
//! the cell to FD; these anchors are the acceptance criterion for that routing.
//!
//! Measured on `tad_lag_A` at the shared parameter point:
//!
//! | route | OFV | Δ vs NONMEM |
//! |---|---|---|
//! | NONMEM | −271.990 | — |
//! | ferx, analytic gradient (pre-fix) | −272.1658 | 0.176 |
//! | ferx, FD gradient (post-fix) | −271.9897 | **0.0003** |
//!
//! ## Why `WT` is in the data with a FIXED-zero exponent
//!
//! ferx routes a subject to its event-driven predictor only when a covariate column
//! actually varies within the subject; the plain (dense) predictor returns `NaN` for
//! a `TAD`-reading RHS under a lagtime (#1110, a separate pre-existing defect —
//! **fixed since #1124**, which routes any model-time-reading RHS to the
//! event-driven predictor directly, so this `WT` trick is no longer required. It
//! is kept only so a passing anchor is not perturbed: removing it drops `THETA(5)`
//! and needs a fresh NONMEM run and a new committed OFV. Do not copy it into a new
//! anchor — `mr_tad.ctl` has no `WT` column). `WT`
//! varies here purely to force that routing, and its exponent is fixed at `0` in
//! **both** engines, so `(WT/70)^0 == 1` exactly and `WT` cannot change any
//! prediction. That also means every per-event covariate snapshot is identical, so
//! this anchor does not exercise the time-varying-covariate snapshot convention and
//! is unaffected by #1073's change to where the timeline breaks.
//!
//! ## Why the control streams look the way they do
//!
//! NONMEM has no `TAD` built-in in `$DES`, so the anchor is carried explicitly. Two
//! constructions were measured **wrong** before the committed one, against an
//! independent stdlib-RK4 reference (`nonmem_anchor/tad_lag_deterministic_truth.py`,
//! run with `$OMEGA 0 FIX` so `IPRED` is a pure function and no EBE can confound
//! it):
//!
//! - `TDP`/`TDN` + `MPAST(1)` with `MTIME` set at every record → **40 % IPRED error**.
//! - the same with `MTIME` set only inside `IF (EVID.EQ.1)` → NM **WARNING 3**
//!   (`MTIME` is zero whenever the condition fails) and the run does not execute.
//!
//! Independently of those, omitting `MTDIFF = 1` while resetting `MTIME` is only
//! **NM WARNING 80**, yet moved the OFV by 41 units. The committed streams derive
//! `TDOS` directly from `T` and use `MTIME` solely to place the integration break at
//! the second arrival; that reproduces the reference to table precision (3.3e-5),
//! and ferx reproduces it to 8.5e-6.
//!
//! ## The kit (`nonmem_anchor/`)
//!
//! - `simulate_tad_lag_data.py` — deterministic generator (pure stdlib, seeds
//!   20260828 / 20260829), 12 subjects. First observation at `t = 2`, chosen so no
//!   record precedes the first arrival for any subject (see #1110).
//! - `tad_lag_A.ctl` / `tad_lag_B.ctl` — single-dose and two-dose controls;
//!   outputs in `results/tad_lag_{A,B}.{ext,lst}`.
//! - `tad_lag_{A,B}_fit.ferx` — the matching ferx models.
//! - `data/tad_lag_{A,B}.csv` — the same CSVs (dose `CMT=1`, obs `CMT=1`, so no
//!   re-keying is needed between engines).

use ferx_core::ode::OdeMethod;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// Shared model text; `{DOSE_COMMENT}` differs only in the doc header.
const MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(20.0, 0.001, 500.0)
  theta THETA_TAD(0.3, 0.0, 10.0)
  theta TVLAG(0.5, 0.001, 10.0)
  theta THETA_WT(0.0, -5.0, 5.0) FIX

  omega ETA_CL ~ 0.09
  omega ETA_LAG ~ 0.02

  sigma PROP ~ 0.1 (sd)

[individual_parameters]
  CL      = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V       = TVV
  KTAD    = THETA_TAD
  LAGTIME = TVLAG * exp(ETA_LAG)

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -(CL/V) * central * (1.0 + KTAD * TAD)

[scaling]
  y = central / V

[covariates]
  WT continuous

[error_model]
  DV ~ proportional(PROP)
";

fn run_anchor(data: &str, nonmem_ofv: f64) {
    const OFV_TOLERANCE: f64 = 0.5;

    let model = parse_full_model(MODEL)
        .expect("TAD + lagtime model must parse")
        .model;
    let pop = read_nonmem_csv(Path::new(data), None, None).expect("anchor data must load");

    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        // Evaluate at the shared parameter point: the NONMEM runs are
        // `MAXEVAL=0 POSTHOC`, so their "optimum" IS the initial vector above.
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        // NONMEM-equivalent ODE accuracy, and the stepper actually pinned:
        // `FitOptions::default()` is `OdeMethod::Auto`, so without this line a
        // change to the stiffness probe could move the anchor, and the committed
        // companion (`nonmem_anchor/*_fit.ferx`, which sets `ode_method = rk45`)
        // and this test would disagree at birth with nothing to catch it.
        ode_method: OdeMethod::Rk45,
        ode_reltol: 1e-9,
        ode_abstol: 1e-11,
        inner_tol: 1e-6,
        ..Default::default()
    };

    let res = fit(&model, &pop, &model.default_params, &opts).expect("fit must evaluate");

    // The gate must actually be what produced this number. Without this the test
    // would still pass if the analytic route were restored and happened to land
    // inside the tolerance on some future dataset.
    assert!(
        {
            let m = res.gradient_method_inner.to_lowercase();
            m.contains("fd") || m.contains("finite")
        },
        "#1070: a TAD-reading RHS under an estimated lagtime must route to FD, \
         but the inner gradient reports `{}`",
        res.gradient_method_inner
    );

    assert!(
        res.ofv.is_finite(),
        "OFV must be finite (a NaN here means #1110 has regressed into this cell)"
    );
    let delta = (res.ofv - nonmem_ofv).abs();
    assert!(
        delta < OFV_TOLERANCE,
        "ferx OFV {} vs NONMEM {nonmem_ofv} — delta {delta} exceeds {OFV_TOLERANCE}",
        res.ofv
    );
}

/// Single dose. The defect is in the segment integration itself, not in a
/// dose-event saltation — the gradient is already wrong at the first observation,
/// before any second dose exists — so one dose genuinely exercises it.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored TAD×lagtime (#1070) acceptance: opt in with --features slow-tests"
)]
fn tad_lag_single_dose_matches_nonmem() {
    run_anchor("data/tad_lag_A.csv", -271.990);
}

/// Two doses, so the `TAD` anchor **switches** at the second arrival — a distinct
/// behaviour from the single-dose case, and one that lands strictly inside a
/// record-to-record interval.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored TAD×lagtime (#1070) acceptance: opt in with --features slow-tests"
)]
fn tad_lag_two_dose_matches_nonmem() {
    run_anchor("data/tad_lag_B.csv", -169.317);
}
