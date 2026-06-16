//! Tier-2 integration tests for NONMEM coded-`RATE` handling at the data-reader
//! boundary (`read_nonmem_csv`); #324.
//!
//! Before #324 a coded `RATE` (`-1`/`-2`) on a dose row silently loaded as a
//! bolus (wrong predictions, no warning). Now: `-2` (modeled infusion *duration*)
//! is accepted and carried as [`ferx_core::RateMode::ModeledDuration`] (the
//! `D{cmt}`/engine check happens later at the model+data join, exercised in
//! `modeled_duration.rs`); `-1` (modeled *rate*, #324 Phase B) is still rejected
//! at import. These return immediately (a read-time error / a single parse, no
//! convergence loop), so they need no `slow-tests` gate and run in the PR job.

use ferx_core::read_nonmem_csv;
use std::io::Write;
use tempfile::NamedTempFile;

fn write_csv(contents: &str) -> NamedTempFile {
    let mut f = NamedTempFile::new().expect("create temp csv");
    f.write_all(contents.as_bytes()).expect("write temp csv");
    f.flush().expect("flush temp csv");
    f
}

#[test]
fn coded_rate_minus_one_is_rejected_via_public_reader() {
    // RATE=-1 = NONMEM "infusion rate modeled (R1)"; unsupported → hard error.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
               1,0,.,1,100,1,-1,1\n\
               1,1,5.0,0,.,1,.,0\n";
    let f = write_csv(csv);
    let err = read_nonmem_csv(f.path(), None, None).unwrap_err();
    assert!(err.contains("RATE=-1") && err.contains("R1"), "{err}");
    assert!(err.contains("subject 1"), "{err}");
}

#[test]
fn coded_rate_minus_two_reads_as_modeled_duration_via_public_reader() {
    // RATE=-2 = NONMEM "infusion duration modeled (D{cmt})". As of #324 this is
    // accepted at read time and carried as ModeledDuration; the D{cmt}/ODE-engine
    // check happens later at the model+data join (see modeled_duration.rs). The
    // dose reports as an infusion (its concrete rate/duration are resolved per
    // iteration from the model parameter).
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
               1,0,.,1,100,1,-2,1\n\
               1,1,5.0,0,.,1,.,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).expect("RATE=-2 reads (modeled duration)");
    let dose = &pop.subjects[0].doses[0];
    assert_eq!(dose.rate_mode, ferx_core::RateMode::ModeledDuration);
    assert!(dose.is_infusion(), "a modeled-duration dose is an infusion");
    assert_eq!(dose.amt, 100.0);
}

#[test]
fn positive_rate_reads_as_infusion_via_public_reader() {
    // The supported form: a positive RATE is a constant-rate infusion with
    // duration = AMT / RATE (= 100 / 50 = 2 h). Must read cleanly.
    let csv = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV\n\
               1,0,.,1,100,1,50,1\n\
               1,1,5.0,0,.,1,.,0\n";
    let f = write_csv(csv);
    let pop = read_nonmem_csv(f.path(), None, None).expect("positive RATE must read");
    let dose = &pop.subjects[0].doses[0];
    assert!(dose.is_infusion(), "positive RATE should be an infusion");
    assert!(
        (dose.duration - 2.0).abs() < 1e-9,
        "duration = {}",
        dose.duration
    );
}
