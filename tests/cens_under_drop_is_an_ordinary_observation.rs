//! #1499, end to end: under the default `bloq_method = drop` a nonzero `CENS`
//! flag must change **nothing** — not the objective, not IWRES, not CWRES.
//!
//! The oracle is the same fit on the same dataset with the flag set to `0`, so
//! nothing here restates an expected number: the property is that `drop` ignores
//! the column, and the twin is what "ignores" means. The measurement on the
//! issue was the other half of this pair — OFV identical to 6 decimals both
//! ways, while the `CENS = 7` row came back with blank IWRES/CWRES and subject
//! 1's *other* CWRES moved from −0.053139 to −0.052511, because dropping a row
//! from `R̃^{-1/2}(y − f0)` moves the rows that stay.
//!
//! `maxiter = 0` makes this one objective evaluation plus the post-fit sweep —
//! Tier 2, no convergence loop.

use ferx_core::{run_model_with_data, FitResult};
use std::io::Write;
use tempfile::NamedTempFile;

const MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method     = focei
  maxiter    = 0
  covariance = false
  npde_nsim  = 50
";

/// Subject 1 row 2 carries the flag; subject 2 is the untouched control, so a
/// change that blanks *every* subject cannot pass on the strength of subject 2.
fn csv(flag: &str) -> String {
    format!(
        "ID,TIME,DV,EVID,AMT,CMT,MDV,CENS\n\
         1,0,.,1,100,1,1,0\n\
         1,1,9.0,0,.,1,0,0\n\
         1,2,8.2,0,.,1,0,{flag}\n\
         1,4,6.7,0,.,1,0,0\n\
         2,0,.,1,100,1,1,0\n\
         2,1,9.4,0,.,1,0,0\n\
         2,2,8.0,0,.,1,0,0\n\
         2,4,6.3,0,.,1,0,0\n"
    )
}

fn temp(contents: &str, suffix: &str) -> NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(suffix)
        .tempfile()
        .expect("temp file");
    f.write_all(contents.as_bytes()).expect("write");
    f.flush().expect("flush");
    f
}

fn fit(data: &str) -> FitResult {
    let m = temp(MODEL, ".ferx");
    let d = temp(data, ".csv");
    let mp = m.path().to_str().expect("utf-8 temp path").to_string();
    let dp = d.path().to_str().expect("utf-8 temp path").to_string();
    run_model_with_data(&mp, Some(&dp))
        .map(|(r, _)| r)
        .expect("fit")
}

/// `1` is the ordinary below-LLOQ flag and `7` the out-of-range value
/// `W_CENS_UNEXPECTED` exists to report; under `drop` neither is censored, so
/// both must reproduce the unflagged fit exactly.
#[test]
fn a_flagged_row_under_drop_matches_the_unflagged_twin() {
    let control = fit(&csv("0"));
    for flag in ["1", "7"] {
        let flagged = fit(&csv(flag));

        assert_eq!(
            flagged.ofv, control.ofv,
            "CENS={flag} under `drop` must not move the objective"
        );
        assert_eq!(flagged.subjects.len(), control.subjects.len());

        for (got, want) in flagged.subjects.iter().zip(control.subjects.iter()) {
            assert!(
                got.iwres.iter().all(|w| w.is_finite()),
                "CENS={flag}: subject {} has a blank IWRES: {:?}",
                got.id,
                got.iwres
            );
            assert!(
                got.cwres.iter().all(|c| c.is_finite()),
                "CENS={flag}: subject {} has a blank CWRES: {:?}",
                got.id,
                got.cwres
            );
            // Not just finite — identical, including the rows that *keep* their
            // value only if the flagged row stayed in the decorrelation.
            assert_eq!(
                got.iwres, want.iwres,
                "CENS={flag}: subject {}'s IWRES differs from the unflagged twin",
                got.id
            );
            assert_eq!(
                got.cwres, want.cwres,
                "CENS={flag}: subject {}'s CWRES differs from the unflagged twin",
                got.id
            );
            // NPD is masked per row and NPDE per *subject*, so a flagged row
            // used to cost its subject every NPDE it had. `npde_nsim = 50`
            // populates both (the reference draws are seeded, so the twin's
            // values are comparable, not merely finite).
            assert!(
                !got.npd.is_empty() && !got.npde.is_empty(),
                "CENS={flag}: subject {} has no simulation-based scores — \
                 `npde_nsim` did not reach this fit",
                got.id
            );
            assert!(
                got.npd.iter().chain(got.npde.iter()).all(|v| v.is_finite()),
                "CENS={flag}: subject {} has a blank NPD/NPDE: npd={:?} npde={:?}",
                got.id,
                got.npd,
                got.npde
            );
            assert_eq!(
                (&got.npd, &got.npde),
                (&want.npd, &want.npde),
                "CENS={flag}: subject {}'s NPD/NPDE differ from the unflagged twin",
                got.id
            );
        }
    }
}

/// The control leg: with `bloq_method = m3` the same flag *does* censor the row,
/// so this pair straddles the gate. Without it, an implementation that ignored
/// `CENS` everywhere would satisfy the test above.
#[test]
fn the_same_row_under_m3_is_censored() {
    let m3_model = MODEL.replace("  covariance = false", "  covariance = false\n  bloq = m3");
    let m = temp(&m3_model, ".ferx");
    let d = temp(&csv("1"), ".csv");
    let mp = m.path().to_str().expect("utf-8 temp path").to_string();
    let dp = d.path().to_str().expect("utf-8 temp path").to_string();
    let m3 = run_model_with_data(&mp, Some(&dp))
        .map(|(r, _)| r)
        .expect("fit");

    let subject1 = &m3.subjects[0];
    assert!(
        subject1.iwres[1].is_nan() && subject1.cwres[1].is_nan(),
        "under `m3` the flagged row is censored: iwres={:?} cwres={:?}",
        subject1.iwres,
        subject1.cwres
    );
    assert!(
        subject1.npd[1].is_nan() && subject1.npde.iter().all(|v| v.is_nan()),
        "under `m3` the row loses its NPD and voids the subject's NPDE: \
         npd={:?} npde={:?}",
        subject1.npd,
        subject1.npde
    );
    // And the objective differs from the `drop` reading of the same file — the
    // two methods score that row by different terms.
    let drop = fit(&csv("1"));
    assert!(
        (m3.ofv - drop.ofv).abs() > 1e-6,
        "m3 and drop must score the flagged row differently: {} vs {}",
        m3.ofv,
        drop.ofv
    );
}
