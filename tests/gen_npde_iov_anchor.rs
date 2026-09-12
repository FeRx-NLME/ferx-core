//! One-shot generator for the **NPDE/NPD + IOV** NONMEM anchor (#734).
//! Run manually:
//!   cargo test --test gen_npde_iov_anchor --no-default-features --features ci,slow-tests -- --nocapture
//!
//! Writes the shared dataset `nonmem_anchor/npde_iov_anchor.csv` and prints the ferx NPD/NPDE scores so the NONMEM control stream
//! `nonmem_anchor/npde_iov_anchor.ctl` (`$TABLE ... NPDE NPD ESAMPLE=`) can be run on the
//! identical data and the two reference distributions compared row by row. The committed
//! comparison lives in `tests/npde_iov_nonmem_anchor.rs`.
//!
//! Design: `EVID=4` (reset + bolus) at the start of each of three occasions, three
//! post-dose observations each (tad 1/4/8 h at ke = 0.1/h, so the profile spans ~0.9 to
//! ~0.45 of the peak and every row is informative). The reset makes the occasions
//! independent, so nothing but the occasion κ varies within a subject's three profiles —
//! the quantity #734 is about. DV is simulated once under the same θ/Ω/Ω_IOV/Σ the
//! diagnostic then uses, so the NPD scores of a correct reference distribution are
//! standard normal by construction.

use ferx_core::{parse_model_file, read_nonmem_csv, simulate_with_seed};
use std::fmt::Write as _;
use std::path::Path;

const OCC_STARTS: [f64; 3] = [0.0, 24.0, 48.0];
const TADS: [f64; 3] = [1.0, 4.0, 8.0];
const N_SUBJECTS: usize = 60;
const DOSE: f64 = 100.0;
/// Seed for the one-replicate DV draw that becomes the committed dataset.
const SIM_SEED: u64 = 734_734;
/// Replicates in the ferx reference distribution. Matches `ESAMPLE` in the `.ctl`.
const NSIM: usize = 2000;
/// Seed for the ferx reference distribution.
const NPDE_SEED: u64 = 734;

/// The design as a NONMEM CSV, with `dv` supplying the DV cell of observation row `i`.
fn design_csv(dv: impl Fn(usize) -> String) -> String {
    let mut csv = String::from("ID,TIME,DV,EVID,AMT,CMT,MDV,OCC\n");
    let mut i = 0usize;
    for id in 1..=N_SUBJECTS {
        for (o, &start) in OCC_STARTS.iter().enumerate() {
            let occ = o + 1;
            // EVID=4: reset the compartment and dose, so each occasion is a fresh profile.
            let _ = writeln!(csv, "{id},{start},.,4,{DOSE},1,1,{occ}");
            for &tad in &TADS {
                let _ = writeln!(csv, "{id},{},{},0,.,1,0,{occ}", start + tad, dv(i));
                i += 1;
            }
        }
    }
    csv
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "manual anchor generator: opt in with --features slow-tests"
)]
fn generate_npde_iov_anchor() {
    let model = parse_model_file(Path::new("tests/fixtures/npde_iov_anchor.ferx"))
        .expect("the anchor model parses");
    assert_eq!(model.n_kappa, 1, "the anchor declares one occasion kappa");

    // Pass 1: the design with placeholder DV, read back so the EVID=4 resets and the
    // occasion labels are materialized exactly as the committed dataset will be.
    std::fs::create_dir_all("nonmem_anchor").expect("nonmem_anchor/ is writable");
    let scratch = Path::new("nonmem_anchor/npde_iov_anchor.design.csv");
    std::fs::write(scratch, design_csv(|_| "1".into())).expect("scratch design written");
    let design = read_nonmem_csv(scratch, None, Some("OCC")).expect("the design loads");
    std::fs::remove_file(scratch).expect("scratch removed");

    // Pass 2: one simulated replicate (BSV η + per-occasion κ + proportional residual)
    // becomes the committed DV.
    let sims = simulate_with_seed(&model, &design, &model.default_params, 1, SIM_SEED);
    assert_eq!(
        sims.len(),
        N_SUBJECTS * OCC_STARTS.len() * TADS.len(),
        "one simulated row per observation record"
    );
    let dv: Vec<f64> = sims.iter().map(|r| r.outcome.continuous_value()).collect();
    assert!(
        dv.iter().all(|v| v.is_finite() && *v > 0.0),
        "simulated DV must be finite and positive"
    );
    let csv = design_csv(|i| format!("{:.6}", dv[i]));
    // One copy only: `nonmem_anchor/` is where the NONMEM control stream reads it
    // from, and `tests/npde_iov_nonmem_anchor.rs` reads the same path, so the two
    // sides cannot drift.
    std::fs::write("nonmem_anchor/npde_iov_anchor.csv", &csv).expect("anchor CSV written");
    println!(
        "WROTE nonmem_anchor/npde_iov_anchor.csv ({N_SUBJECTS} subjects x {} occasions x {} obs)",
        OCC_STARTS.len(),
        TADS.len()
    );

    // The ferx reference distribution on the committed data.
    let pop = read_nonmem_csv(
        Path::new("nonmem_anchor/npde_iov_anchor.csv"),
        None,
        Some("OCC"),
    )
    .expect("the anchor data loads");
    let out = ferx_core::stats::npde::compute_npde_npd(
        &model,
        &pop,
        &model.default_params,
        NSIM,
        Some(NPDE_SEED),
    );

    // One row per observation, in dataset order, for the row-by-row comparison
    // against the NONMEM `$TABLE`.
    let mut tab = String::from("ID TIME NPD NPDE\n");
    let mut all: Vec<f64> = Vec::new();
    for (subj, s) in pop.subjects.iter().zip(&out) {
        for (j, t) in subj.obs_times.iter().enumerate() {
            let _ = writeln!(tab, "{} {} {:.6} {:.6}", subj.id, t, s.npd[j], s.npde[j]);
            all.push(s.npd[j]);
        }
    }
    std::fs::write("nonmem_anchor/npde_iov_anchor.ferx.tab", tab).expect("ferx scores written");
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    let sd = (all.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (all.len() - 1) as f64).sqrt();
    println!("FERX_NPD n = {} mean = {mean:.5} sd = {sd:.5}", all.len());
}
