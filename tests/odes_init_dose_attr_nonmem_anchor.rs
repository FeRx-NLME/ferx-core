//! NONMEM anchor for #1046 — a dose attribute read by an `[odes] init(...)` seed.
//!
//! #993 rejects a dose attribute (`F`, `LAGTIME`/`ALAG`, `F{n}`/`ALAG{n}`) that the
//! `[odes]` **RHS** reads, because folding it into the flux applies it a second time
//! to every gram that ever entered the compartment. That rule was extended to the
//! `init(...)` seed as well, on the argument that a user told to drop `F` from the
//! RHS could otherwise move it into the initial condition and get "the same" double
//! application. They do not, and this file is the measurement that says so.
//!
//! An initial condition is **not a dose**: `OdeSpec::initial_state` writes the raw
//! expression value into the state vector, and `F` / lag are resolved through
//! `DoseAttrMap` at dose events only — a path the seed never takes. So the dose term
//! carries `F` once and the init term carries `F` once, on two *different*
//! quantities. `init(central) = F * 100` — the bioavailable residue of a pre-study
//! 100 mg dose — is a legitimate model, and rejecting it advised the one repair that
//! is always wrong for it: renaming the parameter whose meaning *is* bioavailability.
//!
//! Every other measurement behind that argument is ferx reading its own source. This
//! file closes the gap against NONMEM 7.6.0.
//!
//! # The four control streams
//!
//! `ADVAN13 TOL=9`, one compartment, `$THETA` all `FIX`, `MAXEVAL=0`, `$OMEGA 0 FIX`,
//! 100 mg bolus into compartment 1 at t = 1, `CL = 5`, `V = 50`.
//!
//! * `odes_init_dose_attr_f_A.ctl` — `F1 = 0.5` (scales the dose) **and**
//!   `A_0(1) = F1*100`.
//! * `odes_init_dose_attr_f_B.ctl` — control: `F1 = 0.5` still scales the dose, but
//!   the seed reads `FSEED = 0.5`, an **ordinary** `$PK` variable of the same value.
//! * `odes_init_dose_attr_lag_{A,B}.ctl` — the same pair for `ALAG1 = 0.7`.
//!
//! The A/B split is the point: it holds the *value* fixed and varies only whether the
//! name is a dose attribute, so it isolates the one question — does an initial
//! condition treat `F1`/`ALAG1` as a dose attribute, or as an ordinary number?
//!
//! # Measured (`nonmem_anchor/results/odes_init_dose_attr_*.tab`)
//!
//! ```text
//!    t   F-A / F-B    LAG-A / LAG-B
//!  0.0   1.00000      1.40000
//!  0.5   0.95123      1.33170
//!  1.0   1.90480      1.26680     <- dose record; F: +50, LAG: no jump yet
//!  1.5   1.81190      1.20500
//!  2.0   1.72360      3.08710
//!  4.0   1.41110      2.52750
//!  8.0   0.94591      1.69420
//! ```
//!
//! Three facts, each independently readable off that table:
//!
//! 1. **The seed is the raw value.** `IPRED(0) = 1.0 = (F1·100)/V = 50/50`. Had NONMEM
//!    applied bioavailability to the seed it would read `25/50 = 0.5`. For lag,
//!    `IPRED(0) = 1.4 = (ALAG1·100)/V = 70/50`, deposited at t = 0 **unshifted** — a
//!    lagged seed would leave the compartment empty until t = 0.7.
//! 2. **The dose still gets the attribute, exactly once.** The F run jumps by
//!    `50 = F1·100` at t = 1, and the LAG run shows *no* jump at t = 1 or t = 1.5 —
//!    the dose lands at t = 1.7, which is where the t = 2 value comes from.
//! 3. **A and B are byte-identical** (`.tab` files compare equal, `#OBJV` 575.124 for
//!    the F pair and 579.151 for the lag pair). So the reference engine treats a
//!    dose-attribute name in an initial condition exactly as it treats an ordinary
//!    parameter of the same value — which is precisely ferx's contract.
//!
//! # A NONMEM detail worth recording
//!
//! NONMEM deposits `A_0(n)` at the time of the subject's **first data record**, not at
//! t = 0. An earlier draft of these streams had its first record at t = 0.5 and the
//! seed landed there undecayed, which would have made a ferx comparison mismatch for a
//! reason unrelated to this issue. The committed dataset therefore carries an explicit
//! t = 0 observation, so NONMEM's convention coincides with ferx's (`initial_state` at
//! record start) and the seed is also readable directly as `IPRED(0) = seed/V`.
//!
//! Tier 2: `predict` at fixed parameters, no convergence loop, so no `slow-tests` gate.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv};
use std::io::Write;
use tempfile::NamedTempFile;

/// `nonmem_anchor/odes_init_dose_attr.csv`, keyed for ferx. Identical event
/// structure; `DV` is a placeholder (every prediction here is at fixed parameters).
const CSV: &str = "ID,TIME,DV,MDV,EVID,AMT,CMT\n\
1,0,1.0,0,0,0,1\n\
1,0.5,1.0,0,0,0,1\n\
1,1,.,1,1,100,1\n\
1,1.5,1.0,0,0,0,1\n\
1,2,1.0,0,0,0,1\n\
1,4,1.0,0,0,0,1\n\
1,8,1.0,0,0,0,1\n";

/// ferx equivalent of the control streams. `attr` is the dose-attribute parameter
/// (`F` or `LAGTIME`), `init_reads` the name the `init(...)` seed multiplies — the
/// attribute itself for the A arm, an ordinary parameter for the B arm.
fn model(attr: &str, attr_init: f64, init_reads: &str) -> String {
    format!(
        "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVATTR({attr_init}, 0.0, 1e15)
  theta TVSEED({attr_init}, 0.0, 1e15)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  {attr} = TVATTR
  SEED = TVSEED

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  init(central) = {init_reads} * 100.0
  d/dt(central) = -(CL/V) * central

[scaling]
  obs_scale = V

[error_model]
  DV ~ proportional(PROP)
"
    )
}

/// NONMEM `IPRED` at the six **observation** times (t = 0, 0.5, 1.5, 2, 4, 8); the
/// t = 1 dose record carries no ferx prediction. `F1 = 0.5`, seed `F1·100 = 50`.
const NONMEM_F: [f64; 6] = [
    1.0000E+00, 9.5123E-01, 1.8119E+00, 1.7236E+00, 1.4111E+00, 9.4591E-01,
];

/// The lag pair: `ALAG1 = 0.7`, seed `ALAG1·100 = 70`, dose lands at t = 1.7.
const NONMEM_LAG: [f64; 6] = [
    1.4000E+00, 1.3317E+00, 1.2050E+00, 3.0871E+00, 2.5275E+00, 1.6942E+00,
];

fn preds(src: &str) -> Vec<f64> {
    let mut f = NamedTempFile::new().expect("temp csv");
    write!(f, "{CSV}").expect("write csv");
    f.flush().expect("flush csv");
    let pop = read_nonmem_csv(f.path(), None, None).expect("dataset loads");
    let model = parse_full_model(src).expect("the model parses").model;
    predict(&model, &pop, &model.default_params)
        .into_iter()
        .map(|p| p.pred)
        .collect()
}

fn assert_matches(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: one prediction per observation"
    );
    // 1e-4 relative, matching the #993 anchor: NONMEM's table carries 5 significant
    // figures, so this sits just above the reference's own precision.
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let rel = (g - w).abs() / w;
        assert!(
            rel < 1e-4,
            "{label} obs {i}: ferx {g:.6} vs NONMEM {w:.6} (rel {rel:.2e})"
        );
    }
}

#[test]
fn ferx_matches_nonmem_when_an_init_seed_reads_bioavailability() {
    // The A arm: `F` scales the dose *and* is read by the seed. ferx must accept it
    // (#1046) and reproduce NONMEM — i.e. seed 50, not 25, and a dose scaled once.
    assert_matches(&preds(&model("F", 0.5, "F")), &NONMEM_F, "F-A");
}

#[test]
fn ferx_matches_nonmem_when_an_init_seed_reads_the_absorption_lag() {
    // Same for lag: seed deposited at t=0 unshifted, dose lagged to t=1.7. The t=1.5
    // point is the discriminator — still pure decay, because the dose has not landed.
    assert_matches(
        &preds(&model("LAGTIME", 0.7, "LAGTIME")),
        &NONMEM_LAG,
        "LAG-A",
    );
}

/// Same events as [`CSV`] but with an `EVID=3` reset at t = 1 in place of the dose:
/// the integrator zeroes the state and **re-seeds from `init(...)`**, so the seed
/// expression is evaluated a second time mid-record.
const CSV_RESET: &str = "ID,TIME,DV,MDV,EVID,AMT,CMT\n\
1,0,1.0,0,0,0,1\n\
1,0.5,1.0,0,0,0,1\n\
1,1,.,1,3,0,1\n\
1,1.5,1.0,0,0,0,1\n\
1,2,1.0,0,0,0,1\n";

/// [`CSV_RESET`] with the reset record removed — the same observation times decaying
/// uninterrupted from the t = 0 seed. The contrast that proves the reset re-seeded.
const CSV_NO_RESET: &str = "ID,TIME,DV,MDV,EVID,AMT,CMT\n\
1,0,1.0,0,0,0,1\n\
1,0.5,1.0,0,0,0,1\n\
1,1.5,1.0,0,0,0,1\n\
1,2,1.0,0,0,0,1\n";

#[test]
fn an_init_seed_reading_a_dose_attribute_survives_a_system_reset() {
    // Degenerate oracle (ferx-internal, no NONMEM arm): `initial_state` is used both
    // at record start *and* to re-seed after an `EVID=3/4` reset, so accepting the
    // read means the seed expression is now evaluated on the reset path too. Nothing
    // about a reset consults `DoseAttrMap`, so the attribute must stay an ordinary
    // value there as well — pinned the same way, against the twin that seeds from an
    // ordinary parameter of equal value.
    //
    // Without this the reset path would be an untested inference rather than a
    // measured one; it is the only place the accepted expression runs more than once.
    for (attr, value) in [("F", 0.5), ("LAGTIME", 0.7)] {
        let run = |csv: &str, init_reads: &str| -> Vec<f64> {
            let mut f = NamedTempFile::new().expect("temp csv");
            write!(f, "{csv}").expect("write csv");
            f.flush().expect("flush csv");
            let pop = read_nonmem_csv(f.path(), None, None).expect("dataset loads");
            let m = parse_full_model(&model(attr, value, init_reads))
                .expect("the model parses")
                .model;
            predict(&m, &pop, &m.default_params)
                .into_iter()
                .map(|p| p.pred)
                .collect()
        };
        let a = run(CSV_RESET, attr);
        let b = run(CSV_RESET, "SEED");
        assert_eq!(a.len(), 4, "{attr}: four observations across the reset");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{attr} obs {i}: re-seeding after EVID=3 from the dose attribute ({x}) \
                 must equal re-seeding from an ordinary parameter ({y})"
            );
        }
        // Non-vacuity: the reset must actually re-seed, or the twin above would agree
        // on a trajectory that merely decayed through t = 1 and never exercised the
        // re-seed path at all. Compared against the identical model on a dataset with
        // the reset record removed.
        //
        // Note the post-reset points do *not* differ from their pre-reset
        // counterparts by inspection — t = 1.5 sits 0.5 h after the re-seed exactly as
        // t = 0.5 sits 0.5 h after the original seed, so both read
        // `seed·e^(−k/2)/V` and are bit-identical. That coincidence is *why* this
        // needs the no-reset arm rather than a monotonicity check on `a` alone.
        let plain = run(CSV_NO_RESET, attr);
        assert_eq!(
            plain.len(),
            4,
            "{attr}: four observations without the reset"
        );
        assert_ne!(
            a[2].to_bits(),
            plain[2].to_bits(),
            "{attr}: t=1.5 with the reset ({}) must differ from uninterrupted decay \
             ({}) — otherwise EVID=3 never re-seeded and the twin above is vacuous",
            a[2],
            plain[2]
        );
    }
}

#[test]
fn seeding_from_a_dose_attribute_equals_seeding_from_an_ordinary_parameter() {
    // The A/B control, and the strongest statement of the contract: holding the
    // *value* fixed and varying only whether the seeded name is a dose attribute must
    // change nothing at all. NONMEM's two tables are byte-identical; ferx's must be
    // bit-identical, which is stricter than the 1e-4 the anchor assertions use.
    //
    // This is what makes the anchor a measurement of the *rule* rather than of one
    // model: if the engine ever began applying `F` to the seed, A would diverge from B
    // here even if both still happened to sit inside a loose tolerance of the table.
    for (attr, value) in [("F", 0.5), ("LAGTIME", 0.7)] {
        let a = preds(&model(attr, value, attr));
        let b = preds(&model(attr, value, "SEED"));
        assert_eq!(a.len(), b.len(), "{attr}: same number of predictions");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{attr} obs {i}: seeding from the dose attribute ({x}) must be \
                 bit-identical to seeding from an ordinary parameter of the same \
                 value ({y})"
            );
        }
    }
}
