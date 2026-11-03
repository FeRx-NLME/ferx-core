//! NONMEM anchor for #1898: an analytical model's `[initial_conditions]` amount
//! and Form B `obs_scale` read an individual parameter that no `pk(...)` role
//! binds.
//!
//! Such a name had no PK slot of its own. Both readers look it up through
//! `pk_indices`, whose placeholder for it was `0`, CL's slot, so the init
//! deposited CL and the scale divided by CL, with no warning. The fix gives it
//! a free differentiable slot (`allocate_readout_extra_slots`). This file holds
//! ferx against NONMEM 7.6.0, where the same name is an ordinary `$PK` variable.
//!
//! # The control streams (`nonmem_anchor/unbound_reader_{init,scale}.ctl`)
//!
//! `ADVAN2 TRANS2`, `CL = 5`, `V = 50`, `KA = 1.5`, all `$THETA FIX`, `$OMEGA 0
//! FIX`, `MAXEVAL=0`. A 100 mg oral dose at t = 1. Two subjects, WT 60 and 80, so
//! the read name differs by subject and the read is shown to be live per subject.
//!
//! * `init`: `S = 0.5·WT` (30 / 40), `A_0(2) = S`, `S2 = V`. The t = 0 record
//!   reads the seed directly: `IPRED(0) = S/V` = 0.6 / 0.8.
//! * `scale`: `SCF = 1.0·WT/70`, `S2 = V·SCF`. The two subjects' predictions
//!   differ by exactly 80/60 after the dose (1.1972 / 0.89787 at t = 1.5).
//!   NM-TRAN reserves `SC` as an alias of `S2`, hence the name. Additive error,
//!   because the pre-dose `IPRED` is 0 and a proportional one made the objective
//!   infinite; the error model does not touch `IPRED`.
//!
//! Each dataset carries an explicit t = 0 record: NONMEM deposits `A_0` at the
//! first record, which then coincides with ferx's t = 0 (#1046).
//!
//! # Measured (`nonmem_anchor/results/unbound_reader_{init,scale}.tab`)
//!
//! On `main` (`deaa92f6`) ferx read CL = 5 in both places: an init of 5 instead
//! of 30 / 40, and a scale of 5 instead of 0.857 / 1.143.
//!
//! Tier 2: `predict` at fixed parameters, no convergence loop.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv};
use std::io::Write;
use tempfile::NamedTempFile;

/// `nonmem_anchor/unbound_reader.csv`, verbatim.
const CSV: &str = include_str!("../nonmem_anchor/unbound_reader.csv");

/// The ferx model: `ip` is the read name's definition, `reader` the block
/// that reads it.
fn model(ip: &str, reader: &str) -> String {
    format!(
        "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVKA(1.5, 0.0, 1e15)
  theta TVS(1.0, 0.0, 1e15)
  omega ETA_CL ~ 0.0
  sigma ADD ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  {ip}

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ additive(ADD)

{reader}
"
    )
}

/// NONMEM `IPRED` at the six observation records of each subject (t = 0, 0.5,
/// 1.5, 2, 4, 8; the t = 1 dose record carries no ferx prediction), WT 60 then
/// WT 80.
const NONMEM_INIT: [f64; 12] = [
    6.0000E-01, 5.7074E-01, 1.5426E+00, 1.9520E+00, 1.9659E+00, 1.3336E+00, //
    8.0000E-01, 7.6098E-01, 1.7147E+00, 2.1158E+00, 2.0999E+00, 1.4235E+00,
];
const NONMEM_SCALE: [f64; 12] = [
    0.0, 0.0, 1.1972E+00, 1.7043E+00, 1.8243E+00, 1.2414E+00, //
    0.0, 0.0, 8.9787E-01, 1.2782E+00, 1.3682E+00, 9.3105E-01,
];

fn preds(src: &str) -> Vec<f64> {
    let mut f = NamedTempFile::new().expect("temp csv");
    write!(f, "{CSV}").expect("write csv");
    f.flush().expect("flush csv");
    let pop = read_nonmem_csv(f.path(), Some(&["WT"]), None).expect("dataset loads");
    let model = parse_full_model(src).expect("the model parses").model;
    predict(&model, &pop, &model.default_params)
        .expect("predict")
        .into_iter()
        .map(|p| p.pred)
        .collect()
}

/// NONMEM's table carries 5 significant figures, so 1e-4 relative sits just
/// above the reference's own precision; the pre-dose zeros of the scale arm are
/// compared absolutely. Measured worst error 3.73e-5 relative (init arm)
/// — the table's own rounding, at most 5e-5 — so 1e-4 is 2.7× headroom.
fn assert_matches(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: one prediction per observation"
    );
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(g.is_finite(), "{label} obs {i}: ferx {g}");
        let err = if w == 0.0 { g.abs() } else { (g - w).abs() / w };
        assert!(
            err < 1e-4,
            "{label} obs {i}: ferx {g:.6} vs NONMEM {w:.6} ({err:.2e})"
        );
    }
}

#[test]
fn ferx_matches_nonmem_when_an_init_reads_an_unbound_parameter() {
    let src = model(
        "S = 0.5 * TVS * WT",
        "[initial_conditions]\n  init(central) = S",
    );
    assert_matches(&preds(&src), &NONMEM_INIT, "init");
}

#[test]
fn ferx_matches_nonmem_when_obs_scale_reads_an_unbound_parameter() {
    let src = model("SCF = TVS * WT / 70", "[scaling]\n  obs_scale = SCF");
    assert_matches(&preds(&src), &NONMEM_SCALE, "scale");
}
