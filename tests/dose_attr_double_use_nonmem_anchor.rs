//! NONMEM anchor for the #993 dose-attribute double use.
//!
//! Every other measurement behind #993 is ferx measuring ferx: a model that reads
//! its own `F` versus the same model with the parameter renamed. That is a strong
//! self-consistency check but not an independent one — both sides run on the same
//! engine, so it cannot distinguish "F is applied at the dose *and* in the flux"
//! from any other engine-side explanation giving the same ratio. This file closes
//! that gap against NONMEM 7.6.0.
//!
//! Two `ADVAN13` control streams differing in exactly one line
//! (`nonmem_anchor/dose_attr_double_use_{A,B}.ctl`), `$THETA` all `FIX`,
//! `MAXEVAL=0`, `$OMEGA 0 FIX`:
//!
//! * **A** — modern convention, `F1` applied at the dose only:
//!   `DADT(2) = KA*A(1) - K*A(2)`
//! * **B** — the legacy convention the ODE docs' migration note describes, with
//!   `F1` *also* folded into the absorption flux:
//!   `DADT(2) = F1*KA*A(1) - K*A(2)`
//!
//! Measured (`nonmem_anchor/results/dose_attr_double_use_{A,B}.tab`):
//!
//! ```text
//!    t   NONMEM-A     NONMEM-B     B/A
//!  0.5   0.513070     0.256530     0.499990
//!  1.0   0.730400     0.365200     0.500000
//!  2.0   0.823870     0.411930     0.499994
//!  4.0   0.715540     0.357770     0.500000
//!  8.0   0.481420     0.240710     0.500000
//! 12.0   0.322710     0.161350     0.499985
//! 24.0   0.097198     0.048599     0.500000
//! ```
//!
//! `B/A = 0.5 = F1` at every point (max deviation 1.5e-5, the 5-significant-figure
//! precision of NONMEM's table), so the legacy form really does apply `F` twice —
//! an effective bioavailability of `F²` — and **NONMEM reports nothing**: run B
//! completes and prints an objective function value (646.654 against A's 656.358).
//! ferx being stricter than NONMEM here is therefore a deliberate divergence, not a
//! shared behaviour, which is what `docs/model-file/ode-models.qmd`'s migration note
//! and the ferxtranslate guidance rest on.
//!
//! The two assertions below pin both halves of that:
//!
//! 1. ferx's *accepted* model reproduces NONMEM-A (the correct convention), so
//!    ferx's reading of `F` agrees with the reference implementation.
//! 2. ferx *rejects* the B convention outright, which is the whole of #993.
//!
//! Tier 2: `predict` at fixed parameters plus one parse, no convergence loop, so it
//! needs no `slow-tests` gate.
//!
//! # The analytical half (#1004)
//!
//! #993 shipped the ODE engine only. The same pair re-run on NONMEM's own
//! **analytical** library routine (`ADVAN2 TRANS2`, no `$DES` at all) anchors the
//! analytical rejection — `nonmem_anchor/analytical_dose_attr_double_use_{A,B}.ctl`,
//! differing in exactly one `$PK` line:
//!
//! * **A** — `F1` applied at the dose only, `S2 = V`
//! * **B** — `F1` at the dose *and* in the scale, `S2 = V/F1`
//!
//! This is the analytical spelling of the same defect: `S2` is NONMEM's readout
//! divisor, the exact counterpart of ferx's `[scaling] obs_scale`. Dividing by
//! `V/F1` instead of `V` multiplies the readout by `F1`, so — measured
//! (`nonmem_anchor/results/analytical_dose_attr_double_use_{A,B}.tab`) — `B/A =
//! 0.5 = F1` at every point, digit-for-digit the same table `ADVAN13`'s
//! `F1`-in-the-flux run produced. NONMEM again reports nothing: B completes with
//! `#OBJV = 646.654` against A's `656.358`, and the two `.lst` warning blocks are
//! byte-identical (the NM-TRAN population/ETA boilerplate both runs emit).
//!
//! Note `S1 = V/F` *is* the standard apparent-volume convention when `F1` is **not**
//! separately defined — the `CL/F`, `V/F` parameterisation. What B does is define
//! `F1` and then divide by it again, which double-counts in NONMEM too; NONMEM
//! simply does not say so. Run A also re-derives the #993 `NONMEM_A` table from a
//! different NONMEM routine (closed form vs `ADVAN13`), so the two anchors
//! cross-check each other.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv};
use std::io::Write;
use tempfile::NamedTempFile;

/// `nonmem_anchor/dose_attr_double_use.csv`, keyed for ferx (dose into the depot,
/// observations read from `central` via `obs_cmt`).
const CSV: &str = "ID,TIME,DV,MDV,EVID,AMT,CMT\n\
1,0,.,1,1,100,1\n\
1,0.5,1.0,0,0,0,2\n\
1,1,1.0,0,0,0,2\n\
1,2,1.0,0,0,0,2\n\
1,4,1.0,0,0,0,2\n\
1,8,1.0,0,0,0,2\n\
1,12,1.0,0,0,0,2\n\
1,24,1.0,0,0,0,2\n";

/// ferx equivalent of control stream **A**: `F` declared, applied by the engine at
/// the dose, and absent from the RHS. Same fixed parameters as the `$THETA` block.
const FERX_A: &str = r#"
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVKA(1.5, 0.0, 1e15)
  theta TVF(0.5, 0.0, 1.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = TVF

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[scaling]
  obs_scale = V

[error_model]
  DV ~ proportional(PROP)
"#;

/// NONMEM-A `IPRED` at t = 0.5, 1, 2, 4, 8, 12, 24 h.
const NONMEM_A: [f64; 7] = [
    5.1307E-01, 7.3040E-01, 8.2387E-01, 7.1554E-01, 4.8142E-01, 3.2271E-01, 9.7198E-02,
];

/// NONMEM-B `IPRED` at the same times — the legacy `F1`-in-the-flux convention.
const NONMEM_B: [f64; 7] = [
    2.5653E-01, 3.6520E-01, 4.1193E-01, 3.5777E-01, 2.4071E-01, 1.6135E-01, 4.8599E-02,
];

#[test]
fn ferx_matches_nonmem_when_f_is_applied_only_at_the_dose() {
    let mut f = NamedTempFile::new().expect("temp csv");
    write!(f, "{CSV}").expect("write csv");
    f.flush().expect("flush csv");
    let pop = read_nonmem_csv(f.path(), None, None).expect("dataset loads");
    let model = parse_full_model(FERX_A)
        .expect("the correct convention parses")
        .model;

    let preds: Vec<f64> = predict(&model, &pop, &model.default_params)
        .into_iter()
        .map(|p| p.pred)
        .collect();
    assert_eq!(
        preds.len(),
        NONMEM_A.len(),
        "one prediction per observation"
    );

    // 1e-4 relative: NONMEM's table carries 5 significant figures, and the observed
    // agreement is 8.5e-6, so this has ~10x headroom over the reference's own
    // precision without being loose enough to hide a real divergence.
    for (i, (&got, &want)) in preds.iter().zip(NONMEM_A.iter()).enumerate() {
        let rel = (got - want).abs() / want;
        assert!(
            rel < 1e-4,
            "obs {i}: ferx {got:.6} vs NONMEM {want:.6} (rel {rel:.2e})"
        );
    }
}

#[test]
fn the_convention_nonmem_computes_as_f_squared_is_rejected_by_ferx() {
    // The other half of the anchor. NONMEM runs control stream B without a
    // diagnostic and returns predictions lower by exactly `F1`; ferx refuses to
    // build the model at all. Pin both facts together so neither can drift alone.
    for (i, (&a, &b)) in NONMEM_A.iter().zip(NONMEM_B.iter()).enumerate() {
        let ratio = b / a;
        assert!(
            (ratio - 0.5).abs() < 2e-5,
            "obs {i}: NONMEM B/A = {ratio:.6}, expected F1 = 0.5 (the F² factor)"
        );
    }

    let ferx_b = FERX_A.replace(
        "d/dt(central) =  KA * depot - (CL/V) * central",
        "d/dt(central) =  F * KA * depot - (CL/V) * central",
    );
    assert_ne!(ferx_b, FERX_A, "the B variant must actually differ");
    let err = match parse_full_model(&ferx_b) {
        Err(e) => e,
        Ok(_) => panic!("ferx must reject the F²-in-the-flux convention"),
    };
    assert!(
        err.contains("[odes]:") && err.contains("reserved dose-attribute name"),
        "must be the #993 diagnostic, got: {err}"
    );
}

/// ferx equivalent of the **analytical** control stream A (`ADVAN2 TRANS2`, `F1`
/// applied at the dose only, `S2 = V`): the closed-form `pk one_cpt_oral` block,
/// which divides by `v` internally exactly as NONMEM's implicit `S2 = V` does.
const FERX_ANALYTICAL_A: &str = r#"
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVKA(1.5, 0.0, 1e15)
  theta TVF(0.5, 0.0, 1.0)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = TVF

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)

[error_model]
  DV ~ proportional(PROP)
"#;

/// Analytical NONMEM-B `IPRED` — the `S2 = V/F1` convention, i.e. `F1` applied at
/// the dose and again in the readout scale.
const NONMEM_ANALYTICAL_B: [f64; 7] = [
    2.5653E-01, 3.6520E-01, 4.1193E-01, 3.5777E-01, 2.4071E-01, 1.6135E-01, 4.8599E-02,
];

#[test]
fn ferx_analytical_matches_nonmem_advan2_when_f_is_applied_only_at_the_dose() {
    // Same NONMEM-A table as the ODE anchor above, but produced by a different
    // NONMEM routine (the ADVAN2 closed form rather than ADVAN13's integrator) and
    // reproduced here by ferx's *analytical* engine rather than its ODE engine. So
    // this pins the closed-form path's reading of `F` against the reference
    // implementation's closed-form path, which is the engine #1004 is about.
    let mut f = NamedTempFile::new().expect("temp csv");
    write!(f, "{CSV}").expect("write csv");
    f.flush().expect("flush csv");
    let pop = read_nonmem_csv(f.path(), None, None).expect("dataset loads");
    let model = parse_full_model(FERX_ANALYTICAL_A)
        .expect("the correct analytical convention parses")
        .model;

    let preds: Vec<f64> = predict(&model, &pop, &model.default_params)
        .into_iter()
        .map(|p| p.pred)
        .collect();
    assert_eq!(
        preds.len(),
        NONMEM_A.len(),
        "one prediction per observation"
    );
    for (i, (&got, &want)) in preds.iter().zip(NONMEM_A.iter()).enumerate() {
        let rel = (got - want).abs() / want;
        assert!(
            rel < 1e-4,
            "obs {i}: ferx analytical {got:.6} vs NONMEM ADVAN2 {want:.6} (rel {rel:.2e})"
        );
    }
}

#[test]
fn the_analytical_s_scale_convention_nonmem_computes_as_f_squared_is_rejected_by_ferx() {
    // The #1004 half. NONMEM runs `S2 = V/F1` with `F1` defined and reports
    // nothing, returning predictions lower by exactly `F1`; ferx now refuses to
    // build the analytical equivalent. Pin both facts together so neither drifts.
    for (i, (&a, &b)) in NONMEM_A.iter().zip(NONMEM_ANALYTICAL_B.iter()).enumerate() {
        let ratio = b / a;
        assert!(
            (ratio - 0.5).abs() < 2e-5,
            "obs {i}: NONMEM ADVAN2 B/A = {ratio:.6}, expected F1 = 0.5 (the F² factor)"
        );
    }

    // `obs_scale` is ferx's `S2`: dividing the readout by `F` on top of the
    // engine's dose-time application is the same double count as `S2 = V/F1`.
    // (`obs_scale` is divisive and the `pk` block already divides by `v`, so
    // `1.0 / F` here is the `V/F1` scale — a bare `V` would be the *volume*
    // double-divide, a different, warned-about defect.)
    let ferx_b = format!("{FERX_ANALYTICAL_A}\n[scaling]\n  obs_scale = 1.0 / F\n");
    let err = match parse_full_model(&ferx_b) {
        Err(e) => e,
        Ok(_) => panic!("ferx must reject the analytical F²-in-the-scale convention"),
    };
    assert!(
        err.contains("[scaling]:") && err.contains("remove the `f=F` mapping"),
        "must be the #1004 diagnostic, got: {err}"
    );
}
