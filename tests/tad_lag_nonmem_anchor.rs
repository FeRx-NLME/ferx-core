//! NONMEM cross-check for an `[odes]` RHS that reads the `TAD` built-in while the
//! model carries an **estimated lagtime** — the cell #1070 is about.
//!
//! ## What is being anchored
//!
//! `eval_rhs_anchored` used to compute `tad = t − last_dose_eff` in `f64`, and
//! `eval_rhs_g` wrote it into the RHS variable table as a constant. But
//! `last_dose_eff` **is** the dose's lagged arrival (`d.time + lag`), so
//! `∂TAD/∂lag = −1` never entered the dual chain: the analytic η/θ gradient on the
//! lag axis was wrong everywhere the trajectory is integrated, not merely at an
//! event boundary. The *value* stayed exact, which is why no prediction test ever
//! caught it. `tad` is now threaded as a dual, so the term is carried.
//!
//! Since FOCEI builds its `h` matrix from that same analytic `∂f/∂η`
//! (`inner_optimizer.rs`), the error landed in the **reported OFV** — so it is
//! NONMEM-anchorable, and that is what these tests do. They were first committed as
//! the acceptance criterion for #1070's interim FD routing; they are now the
//! acceptance criterion for the analytic route itself, which is the stronger claim
//! (FD was only ever *avoiding* the wrong chain — this asserts the chain is right).
//!
//! Measured at the shared parameter point:
//!
//! | route | `tad_lag_A` | Δ | `tad_lag_B` | Δ |
//! |---|---|---|---|---|
//! | NONMEM | −271.990 | — | −169.317 | — |
//! | ferx analytic, `tad` lifted as `f64` | −272.1658 | 0.176 | −169.4240 | 0.107 |
//! | ferx FD (the interim #1070 route) | −271.9897 | 0.0003 | −169.3480 | 0.032 |
//! | **ferx analytic, `tad` threaded as a dual** | **−271.99037** | **0.0004** | **−169.34856** | **0.032** |
//!
//! The analytic route now reproduces the FD route to the digits printed here, which
//! is the point: the two independent gradient paths agree, and both agree with
//! NONMEM. The residual 0.032 on `B` belongs to the NONMEM control stream (see the
//! `MTIME` discussion below), not to either ferx route.
//!
//! ## Why `WT` is in the data with a FIXED-zero exponent
//!
//! ferx routes a subject to its event-driven predictor only when a covariate column
//! actually varies within the subject. `WT` varies here purely to force that routing,
//! and its exponent is fixed at `0` in **both** engines, so `(WT/70)^0 == 1` exactly
//! and `WT` cannot change any prediction. That also means every per-event covariate
//! snapshot is identical, so this anchor does not exercise the time-varying-covariate
//! snapshot convention and is unaffected by #1073's change to where the timeline
//! breaks.
//!
//! The routing was originally needed because the plain (dense) predictor answered
//! `NaN` for a `TAD`-reading RHS under a lagtime. **#1073 fixed that** — both ODE
//! predictors now anchor the pre-arrival window at the subject's first arrival, so
//! the dense path is finite too and this anchor would hold on either route. The
//! forcing is kept as belt-and-braces: it pins the engine the measurement was made
//! on, and re-running it after a routing change is a deliberate act rather than an
//! accident. #1110 still owns what `TAD` *means* in that window.
//!
//! **#1124** then made the forcing redundant a second way: a model-time-reading
//! `[odes]` RHS now routes to the event-driven predictor on the predicate itself,
//! with no covariate needed. Removing `WT` would drop `THETA(5)` and needs a fresh
//! NONMEM run and a new committed OFV, so it stays here — but do not copy the trick
//! into a new anchor; `mr_tad.ctl` has no `WT` column.
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
//!   record precedes the first arrival for any subject (see #1110). That also keeps
//!   the anchor independent of the pre-arrival `TAD` value #1073 settled: the
//!   integrator does traverse `(0, ALAG]`, but the compartment is empty there, so the
//!   `TAD` term multiplies zero.
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

/// `tolerance` is per anchor, deliberately, because the two have margins three orders apart
/// and a shared constant would be set by the looser one.
///
/// Anchor A carries no `MTIME` and agrees to 3.7e-4, so it can be pinned tightly and is what
/// actually discriminates the analytic chain from the broken one (the pre-#1070 route was
/// 0.176 off here). Anchor B's residual 0.032 is a *systematic* offset belonging to the NONMEM
/// control stream's `MTIME` construction, not to ferx — against an independent stdlib-RK4
/// reference ferx matches to 8.5e-6 while the control stream's table print resolves to 3.3e-5
/// (see the module docs). Pinning B within 2x of a known systematic offset would make it
/// brittle to any unrelated stepper or `inner_tol` change, so it gets its own, looser bound
/// that still rejects its own pre-#1070 delta of 0.107.
fn run_anchor(data: &str, nonmem_ofv: f64, tolerance: f64) {
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

    // The analytic route must actually be what produced this number. Without this the
    // test would still pass if the dual `tad` were reverted and the model fell back to
    // finite differences — FD is also right here, so only the reported route separates
    // "the analytic chain carries the lagtime term" from "something quietly stopped
    // claiming the model" (#1070).
    assert!(
        res.gradient_method_inner
            .to_lowercase()
            .contains("analytic"),
        "#1070: a TAD-reading RHS under an estimated lagtime must be served analytically, \
         but the inner gradient reports `{}`",
        res.gradient_method_inner
    );

    // ...and the per-subject route, which the line above CANNOT see. The two directions are
    // not symmetric: `gradient_method_inner` is a model-level union
    // (`inner_reports_analytic_model`), so the assertion this replaced — "must be FD" — was
    // sound, because a model-level FD report implies FD for every subject. "Must be analytic"
    // is not: any per-subject decline reaching these 12 subjects would send the fit to FD,
    // which is also right here (see the table above), so the OFV assertion would still pass
    // while the model-level string still read "analytic" and this anchor quietly stopped
    // exercising the analytic chain at all. The FD-fallback warning IS per-subject.
    let fd_fallbacks: Vec<&String> = res
        .warnings
        .iter()
        .filter(|w| {
            let l = w.to_lowercase();
            l.contains("finite difference") || l.contains("fd fallback") || l.contains("falls back")
        })
        .collect();
    assert!(
        fd_fallbacks.is_empty(),
        "#1070: no subject may fall back to finite differences, but the fit warned: {fd_fallbacks:?}"
    );

    assert!(
        res.ofv.is_finite(),
        "OFV must be finite (a NaN here means #1110 has regressed into this cell)"
    );
    let delta = (res.ofv - nonmem_ofv).abs();
    assert!(
        delta < tolerance,
        "ferx OFV {} vs NONMEM {nonmem_ofv} — delta {delta} exceeds {tolerance}",
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
    // A has no `MTIME`; 1e-3 still rejects its pre-#1070 delta of 0.176 by 100x.
    run_anchor("data/tad_lag_A.csv", -271.990, 1e-3);
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
    // B carries the control stream's systematic 0.032; 0.06 rejects its pre-#1070
    // delta of 0.107 while leaving room for unrelated numerical drift.
    run_anchor("data/tad_lag_B.csv", -169.317, 0.06);
}
