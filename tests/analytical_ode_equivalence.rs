//! Equivalence between the analytical PK closed forms and their amount-based
//! ODE transcription (ferx-r issue #127: "standard models - analytical + ode
//! form").
//!
//! For each of the six standard [`PkModel`] types we build an analytical model
//! (`pk <model>(...)`) and an ODE model that is its hand transcription, then
//! `predict()` both on identical populations spanning every dosing mode -
//! bolus, infusion, multiple doses, steady state, lag time and (for oral
//! models) bioavailability - and assert the population PRED agrees pointwise.
//!
//! `predict()` evaluates at eta = 0, so this is a pure check of the *structural*
//! transformation. The transcription rules verified here are:
//!   - ODE states carry **amounts**; the observed concentration is read out via
//!     `[scaling] obs_scale = V` (or `V1` for multi-compartment models) -
//!     equivalent to NONMEM's `S2 = V`.
//!   - inter-compartmental flux uses micro-constants `k10 = CL/V1`,
//!     `k12 = Q2/V1`, `k21 = Q2/V2`, `k13 = Q3/V1`, `k31 = Q3/V3`.
//!   - absorption adds a `depot` state (`-KA*depot` out, `+KA*depot` into
//!     central).
//!   - **bioavailability `F` and lag time are declared as individual
//!     parameters and applied by the engine at the dose** (reserved PK slots) -
//!     they are *never* baked into the `[odes]` RHS. Baking `F` into the flux
//!     would double-count it, since the engine already loads the dose with
//!     `F*AMT` (issue ferx-core #122).
//!
//! Runs on the default/CI feature set (no autodiff) and is fast (no `fit()`),
//! so it is not gated behind `slow-tests` and is run explicitly on every PR
//! from `ci.yml` (not just nightly). Each of the six models is its own
//! `#[test]` (see the `equiv_test!` macro at the bottom) so a regression is
//! isolated to one model and a single case can be run with `cargo test <model>`.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::predict;
use ferx_core::types::{DoseEvent, Population, Subject};

mod common;

/// The RK45 solver defaults to `abstol = 1e-6`, `reltol = 1e-4`
/// (`src/ode/solver.rs`), so the ODE path can only reproduce the exact closed
/// form to about that level. Use a combined absolute+relative tolerance so
/// near-zero predictions (oral early times, multi-dose troughs) don't blow up a
/// pure relative check.
const ATOL: f64 = 1e-5;
const RTOL: f64 = 1e-4;
/// Long trajectories accumulate the solver's per-step error across many dose
/// restarts (multidose out to t=72 over three doses) or the SS iteration, so
/// the ODE path drifts past the per-step `reltol` (1e-4). `RTOL` alone equals
/// that per-step target and is too tight for these cases — give them a looser
/// combined bound so a faithful transcription doesn't flake.
const ACCUM_RTOL: f64 = 1e-3;
/// Relative tolerance for the fixed-parameter OFV (population joint-NLL)
/// equivalence check. The NLL amplifies the per-point prediction error (a tight
/// proportional weight times the residual), so a ~1e-4 prediction agreement
/// shows up as a few × 1e-4 relative NLL agreement, worst at the IV bolus spike.
const OFV_RTOL: f64 = 2e-3;

/// One standard analytical model and the pieces needed to build its ODE twin.
struct Family {
    label: &'static str,
    /// `[parameters]` theta declarations (one per structural parameter).
    thetas: &'static str,
    /// `[individual_parameters]` body (shared verbatim by both forms).
    indiv: &'static str,
    /// Analytical `pk <model>(...` call with the closing paren omitted so
    /// `f=`/`lagtime=` mappings can be appended.
    pk_open: &'static str,
    /// ODE `[structural_model]` line.
    ode_struct: &'static str,
    /// ODE `[odes]` body.
    odes: &'static str,
    /// `[scaling] obs_scale = <expr>` right-hand side (`V` or `V1`).
    obs_scale: &'static str,
    is_oral: bool,
}

impl Family {
    /// The dose always enters compartment 1 — central for IV, depot for oral.
    const DOSE_CMT: usize = 1;

    /// Observed (central) compartment: index 2 for oral (behind the depot),
    /// index 1 for IV. Derived from the route so the pair can never drift apart.
    fn obs_cmt(&self) -> usize {
        if self.is_oral {
            2
        } else {
            1
        }
    }
}

fn fam_one_cpt_iv() -> Family {
    Family {
        label: "one_cpt_iv",
        thetas: r"  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
",
        indiv: r"  CL = TVCL * exp(ETA_CL)
  V  = TVV
",
        pk_open: "pk one_cpt_iv(cl=CL, v=V",
        ode_struct: "ode(obs_cmt=central, states=[central])",
        odes: r"  d/dt(central) = -(CL/V) * central
",
        obs_scale: "V",
        is_oral: false,
    }
}

fn fam_one_cpt_oral() -> Family {
    Family {
        label: "one_cpt_oral",
        thetas: r"  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.2, 0.01, 50.0)
",
        indiv: r"  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
",
        pk_open: "pk one_cpt_oral(cl=CL, v=V, ka=KA",
        ode_struct: "ode(obs_cmt=central, states=[depot, central])",
        odes: r"  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
",
        obs_scale: "V",
        is_oral: true,
    }
}

fn fam_two_cpt_iv() -> Family {
    Family {
        label: "two_cpt_iv",
        thetas: r"  theta TVCL(3.0, 0.01, 100.0)
  theta TVV1(15.0, 1.0, 500.0)
  theta TVQ(3.0, 0.01, 100.0)
  theta TVV2(30.0, 1.0, 500.0)
",
        indiv: r"  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q  = TVQ
  V2 = TVV2
",
        pk_open: "pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2",
        ode_struct: "ode(obs_cmt=central, states=[central, periph])",
        odes: r"  d/dt(central) = -(CL/V1 + Q/V1) * central + (Q/V2) * periph
  d/dt(periph)  =  (Q/V1) * central - (Q/V2) * periph
",
        obs_scale: "V1",
        is_oral: false,
    }
}

fn fam_two_cpt_oral() -> Family {
    Family {
        label: "two_cpt_oral",
        thetas: r"  theta TVCL(3.0, 0.01, 100.0)
  theta TVV1(15.0, 1.0, 500.0)
  theta TVQ(3.0, 0.01, 100.0)
  theta TVV2(30.0, 1.0, 500.0)
  theta TVKA(1.1, 0.01, 50.0)
",
        indiv: r"  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q  = TVQ
  V2 = TVV2
  KA = TVKA
",
        pk_open: "pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA",
        ode_struct: "ode(obs_cmt=central, states=[depot, central, periph])",
        odes: r"  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V1 + Q/V1) * central + (Q/V2) * periph
  d/dt(periph)  =  (Q/V1) * central - (Q/V2) * periph
",
        obs_scale: "V1",
        is_oral: true,
    }
}

fn fam_three_cpt_iv() -> Family {
    Family {
        label: "three_cpt_iv",
        thetas: r"  theta TVCL(3.0, 0.01, 100.0)
  theta TVV1(15.0, 1.0, 500.0)
  theta TVQ2(3.0, 0.01, 100.0)
  theta TVV2(30.0, 1.0, 500.0)
  theta TVQ3(1.5, 0.01, 100.0)
  theta TVV3(60.0, 1.0, 999.0)
",
        indiv: r"  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
",
        pk_open: "pk three_cpt_iv(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3",
        ode_struct: "ode(obs_cmt=central, states=[central, periph1, periph2])",
        odes: r"  d/dt(central)  = -(CL/V1 + Q2/V1 + Q3/V1) * central + (Q2/V2) * periph1 + (Q3/V3) * periph2
  d/dt(periph1)  =  (Q2/V1) * central - (Q2/V2) * periph1
  d/dt(periph2)  =  (Q3/V1) * central - (Q3/V3) * periph2
",
        obs_scale: "V1",
        is_oral: false,
    }
}

fn fam_three_cpt_oral() -> Family {
    Family {
        label: "three_cpt_oral",
        thetas: r"  theta TVCL(3.0, 0.01, 100.0)
  theta TVV1(15.0, 1.0, 500.0)
  theta TVQ2(3.0, 0.01, 100.0)
  theta TVV2(30.0, 1.0, 500.0)
  theta TVQ3(1.5, 0.01, 100.0)
  theta TVV3(60.0, 1.0, 999.0)
  theta TVKA(1.1, 0.01, 50.0)
",
        indiv: r"  CL = TVCL * exp(ETA_CL)
  V1 = TVV1
  Q2 = TVQ2
  V2 = TVV2
  Q3 = TVQ3
  V3 = TVV3
  KA = TVKA
",
        pk_open: "pk three_cpt_oral(cl=CL, v1=V1, q2=Q2, v2=V2, q3=Q3, v3=V3, ka=KA",
        ode_struct: "ode(obs_cmt=central, states=[depot, central, periph1, periph2])",
        odes: r"  d/dt(depot)    = -KA * depot
  d/dt(central)  =  KA * depot - (CL/V1 + Q2/V1 + Q3/V1) * central + (Q2/V2) * periph1 + (Q3/V3) * periph2
  d/dt(periph1)  =  (Q2/V1) * central - (Q2/V2) * periph1
  d/dt(periph2)  =  (Q3/V1) * central - (Q3/V3) * periph2
",
        obs_scale: "V1",
        is_oral: true,
    }
}

/// Build the analytical and ODE source for a family, optionally adding lag time
/// and/or bioavailability. The two forms share the same `[parameters]` and
/// `[individual_parameters]`; only the structural section differs. Lag/F enter
/// the analytical form through the `pk(...)` mapping and the ODE form purely by
/// declaring a `LAGTIME`/`F` individual parameter (the engine applies both at
/// the dose).
fn build_pair(f: &Family, lag: bool, fbio: bool) -> (String, String) {
    let mut thetas = String::from(f.thetas);
    let mut indiv = String::from(f.indiv);
    let mut pk_extra = String::new();
    if lag {
        thetas.push_str("  theta TVLAG(0.5, 0.0, 5.0)\n");
        indiv.push_str("  LAGTIME = TVLAG\n");
        pk_extra.push_str(", lagtime=LAGTIME");
    }
    if fbio {
        thetas.push_str("  theta TVF(0.7, 0.01, 1.0)\n");
        indiv.push_str("  F = TVF\n");
        pk_extra.push_str(", f=F");
    }

    let header = format!(
        "[parameters]\n{thetas}  omega ETA_CL ~ 0.09\n  sigma PROP ~ 0.01 (sd)\n\n\
         [individual_parameters]\n{indiv}\n"
    );

    let analytical = format!(
        "{header}[structural_model]\n  {pk}{pk_extra})\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n",
        pk = f.pk_open
    );

    let ode = format!(
        "{header}[structural_model]\n  {st}\n\n[odes]\n{odes}\n\
         [scaling]\n  obs_scale = {scale}\n\n[error_model]\n  DV ~ proportional(PROP)\n",
        st = f.ode_struct,
        odes = f.odes,
        scale = f.obs_scale
    );

    (analytical, ode)
}

fn subject(doses: Vec<DoseEvent>, obs_times: Vec<f64>, obs_cmt: usize) -> Subject {
    let n = obs_times.len();
    // `obs_cmts` is carried for realism but inert for the models under test: the
    // ODE twin selects its readout from the `obs_cmt=central` structural
    // directive (`OdeReadout::ObsCmt`) and the analytical path always reads
    // central, so neither side consults `subject.obs_cmts` here. It would only
    // matter under a `PerCmt` readout. So this test does *not* by itself verify
    // per-observation compartment routing.
    common::subject("1", doses, obs_times, vec![0.0; n], vec![obs_cmt; n])
}

fn population(doses: Vec<DoseEvent>, obs_times: Vec<f64>, obs_cmt: usize) -> Population {
    Population {
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![subject(doses, obs_times, obs_cmt)],
    }
}

/// Parse both forms, `predict()` them on `pop`, and assert pointwise agreement.
fn assert_equiv(label: &str, analytical_src: &str, ode_src: &str, pop: &Population, rtol: f64) {
    let an = parse_full_model(analytical_src)
        .unwrap_or_else(|e| panic!("[{label}] analytical model did not parse: {e}"))
        .model;
    let ode = parse_full_model(ode_src)
        .unwrap_or_else(|e| panic!("[{label}] ODE model did not parse: {e}"))
        .model;

    let pa = predict(&an, pop, &an.default_params);
    let po = predict(&ode, pop, &ode.default_params);
    assert_eq!(pa.len(), po.len(), "[{label}] prediction count mismatch");
    assert!(!pa.is_empty(), "[{label}] produced no predictions");

    // Both models `predict()` the same population, so results come back in the
    // same order — pairing by index is valid. (A literal `x.time == y.time`
    // check would be vacuous: both derive `time` from the identical `obs_times`.)
    for (x, y) in pa.iter().zip(po.iter()) {
        let tol = ATOL + rtol * x.pred.abs();
        assert!(
            (x.pred - y.pred).abs() <= tol,
            "[{label}] t={:.3}: analytical PRED {:.6} vs ODE PRED {:.6} (|diff| {:.2e} > tol {:.2e})",
            x.time,
            x.pred,
            y.pred,
            (x.pred - y.pred).abs(),
            tol
        );
    }
}

/// Exercise every dosing mode for one model family.
fn run_family(f: &Family) {
    // Structural equivalence at the likelihood level: same OFV at fixed params,
    // including eta != 0 (a stronger check than the eta=0 PRED sweep below).
    assert_ofv_equiv(f);

    let obs = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let dc = Family::DOSE_CMT;
    let oc = f.obs_cmt();

    // -- base model (no lag, no F) across dosing modes ----------------------
    let (an, ode) = build_pair(f, false, false);

    // single bolus
    assert_equiv(
        &format!("{}/bolus", f.label),
        &an,
        &ode,
        &population(
            vec![DoseEvent::new(0.0, 100.0, dc, 0.0, false, 0.0)],
            obs.clone(),
            oc,
        ),
        RTOL,
    );

    // multiple doses (q24h x3)
    assert_equiv(
        &format!("{}/multidose", f.label),
        &an,
        &ode,
        &population(
            vec![
                DoseEvent::new(0.0, 100.0, dc, 0.0, false, 0.0),
                DoseEvent::new(24.0, 100.0, dc, 0.0, false, 0.0),
                DoseEvent::new(48.0, 100.0, dc, 0.0, false, 0.0),
            ],
            vec![1.0, 6.0, 12.0, 25.0, 30.0, 49.0, 54.0, 72.0],
            oc,
        ),
        // Long trajectory across three dose restarts — solver error
        // accumulates past the per-step `reltol`, like the SS case.
        ACCUM_RTOL,
    );

    // steady state (II = 24)
    assert_equiv(
        &format!("{}/steady_state", f.label),
        &an,
        &ode,
        &population(
            vec![DoseEvent::new(0.0, 100.0, dc, 0.0, true, 24.0)],
            vec![1.0, 4.0, 8.0, 12.0, 23.0],
            oc,
        ),
        ACCUM_RTOL,
    );

    // infusion (IV only - infusion into a depot is not a standard combo)
    if !f.is_oral {
        assert_equiv(
            &format!("{}/infusion", f.label),
            &an,
            &ode,
            &population(
                vec![DoseEvent::new(0.0, 100.0, dc, 50.0, false, 0.0)],
                obs.clone(),
                oc,
            ),
            RTOL,
        );
    }

    // -- lag time -----------------------------------------------------------
    let (an_l, ode_l) = build_pair(f, true, false);
    assert_equiv(
        &format!("{}/lagtime", f.label),
        &an_l,
        &ode_l,
        // 0.75 h sits just past the 0.5 h lag, so the early-rise region right
        // after arrival is checked — not only the late curve. A lag applied
        // in the wrong direction would show up here.
        &population(
            vec![DoseEvent::new(0.0, 100.0, dc, 0.0, false, 0.0)],
            vec![0.75, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
            oc,
        ),
        RTOL,
    );

    // -- bioavailability (every route, IV included since #327) --------------
    // Bolus F applies to the oral depot and the IV bolus alike: the analytical
    // superposition path now scales the dose by F to match the ODE engine
    // (which auto-applies F at the dose). Before #327 the analytical IV/infusion
    // arms dropped F, so this check ran oral-only and the IV/infusion drift went
    // unnoticed.
    let (an_f, ode_f) = build_pair(f, false, true);
    assert_equiv(
        &format!("{}/bioavailability_bolus", f.label),
        &an_f,
        &ode_f,
        &population(
            vec![DoseEvent::new(0.0, 100.0, dc, 0.0, false, 0.0)],
            obs.clone(),
            oc,
        ),
        RTOL,
    );

    // Infusion F (IV families only — infusion into a depot is not a standard
    // combo, mirroring the base-infusion case above). F scales the infusion
    // rate; the duration is preserved, matching NONMEM's F1 on a modeled-rate
    // infusion.
    if !f.is_oral {
        assert_equiv(
            &format!("{}/bioavailability_infusion", f.label),
            &an_f,
            &ode_f,
            &population(
                vec![DoseEvent::new(0.0, 100.0, dc, 50.0, false, 0.0)],
                obs.clone(),
                oc,
            ),
            RTOL,
        );
    }
}

/// One `#[test]` per model family, so a regression names the offending model
/// and `cargo test <model>` runs just that one.
macro_rules! equiv_test {
    ($test:ident, $ctor:ident) => {
        #[test]
        fn $test() {
            run_family(&$ctor());
        }
    };
}

equiv_test!(one_cpt_iv, fam_one_cpt_iv);
equiv_test!(one_cpt_oral, fam_one_cpt_oral);
equiv_test!(two_cpt_iv, fam_two_cpt_iv);
equiv_test!(two_cpt_oral, fam_two_cpt_oral);
equiv_test!(three_cpt_iv, fam_three_cpt_iv);
equiv_test!(three_cpt_oral, fam_three_cpt_oral);

/// Fixed-parameter OFV equivalence: at identical params and a *fixed* eta (no
/// optimizer), the analytical and ODE forms must produce the same population
/// joint NLL — the deterministic inner objective the EBE loop minimizes.
/// Evaluated at eta = 0 and eta != 0 so the random-effect path is exercised,
/// not just the typical-value structure.
///
/// Why not compare a converged `fit()` OFV instead? Because that conflates
/// structural equivalence with optimizer path and ODE-solver noise: two fits of
/// equivalent models from the same start can stop at different points on the
/// (identical) likelihood surface and report OFVs differing by whole units. A
/// fixed-parameter evaluation isolates the thing we actually want to guarantee.
fn assert_ofv_equiv(f: &Family) {
    use ferx_core::stats::likelihood::individual_nll;

    let dc = Family::DOSE_CMT;
    let oc = f.obs_cmt();
    let obs_t = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let (an_src, ode_src) = build_pair(f, false, false);
    let an = parse_full_model(&an_src).unwrap().model;
    let ode = parse_full_model(&ode_src).unwrap().model;

    // DV from the analytical PRED at eta=0, scaled per subject so the NLL has
    // real (non-degenerate) residuals to weigh.
    let dose = || vec![DoseEvent::new(0.0, 100.0, dc, 0.0, false, 0.0)];
    let base = population(dose(), obs_t.clone(), oc);
    let preds = predict(&an, &base, &an.default_params);
    let mut subjects = Vec::new();
    for (i, fac) in [0.85_f64, 1.0, 1.15].into_iter().enumerate() {
        let mut s = subject(dose(), obs_t.clone(), oc);
        s.id = format!("{}", i + 1);
        s.observations = preds.iter().map(|p| (p.pred * fac).max(1e-6)).collect();
        subjects.push(s);
    }
    let pop = Population { subjects, ..base };

    let pop_nll = |m: &ferx_core::CompiledModel, eta: f64| -> f64 {
        let p = &m.default_params;
        pop.subjects
            .iter()
            .map(|s| individual_nll(m, s, &p.theta, &[eta], &p.omega, &p.sigma.values))
            .sum::<f64>()
    };

    for eta in [0.0, 0.3, -0.3] {
        let a = pop_nll(&an, eta);
        let o = pop_nll(&ode, eta);
        let rel = (a - o).abs() / a.abs().max(1.0);
        assert!(
            rel <= OFV_RTOL,
            "[{}] OFV mismatch at eta={eta}: analytical {a:.6} vs ODE {o:.6} \
             (rel {rel:.2e} > {OFV_RTOL:.0e})",
            f.label
        );
    }
}

/// Covariate-on-ODE equivalence. The model-family sweep above carries no
/// covariates, so covariate handling in the ODE path was otherwise guarded only
/// by the ferx-r example test (`two_cpt_oral_cov`). Covariates enter both forms
/// identically — through the shared `[individual_parameters]` block — so an
/// analytical model and its ODE twin must still agree pointwise once a covariate
/// actually moves a parameter. Here WT scales CL on a one-compartment oral
/// model; two subjects with different WT ensure the covariate term is exercised
/// rather than cancelling to the typical value.
#[test]
fn covariate_model_equiv() {
    // Shared params + indiv block (WT on CL); both forms parse the same header.
    let header = "\
[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.2, 0.01, 50.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * exp(ETA_CL)
  V  = TVV
  KA = TVKA
";

    let analytical = format!(
        "{header}\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\n\
         [covariates]\n  WT continuous\n\n[error_model]\n  DV ~ proportional(PROP)\n"
    );
    let ode = format!(
        "{header}\n[structural_model]\n  ode(obs_cmt=central, states=[depot, central])\n\n\
         [odes]\n  d/dt(depot)   = -KA * depot\n  d/dt(central) =  KA * depot - (CL/V) * central\n\n\
         [scaling]\n  obs_scale = V\n\n[covariates]\n  WT continuous\n\n\
         [error_model]\n  DV ~ proportional(PROP)\n"
    );

    // Oral single dose into depot (cmt 1); observe central (cmt 2).
    let obs = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let mk = |id: &str, wt: f64| {
        let mut s = subject(
            vec![DoseEvent::new(
                0.0,
                100.0,
                Family::DOSE_CMT,
                0.0,
                false,
                0.0,
            )],
            obs.clone(),
            2,
        );
        s.id = id.into();
        s.covariates.insert("WT".into(), wt);
        s
    };
    let pop = Population {
        covariate_names: vec!["WT".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects: vec![mk("1", 55.0), mk("2", 95.0)],
    };

    // Single dose out to 24 h is in the per-step `reltol` regime, so `RTOL`.
    assert_equiv("one_cpt_oral_cov/bolus", &analytical, &ode, &pop, RTOL);
}
