//! #2010: NONMEM anchor for two observations at one `TIME` that the reader splits one ULP apart.
//!
//! When the first observation of a tie is followed by an `EVID=2` row (#1810) or a dose (#614),
//! the reader moves it to `TIME.next_down()`, so the subject's observation times hold `[4⁻, 4]`.
//! The dense ODE engines used to dedup their segment `saveat` at `1e-15`, which merges that pair
//! for any tie at `TIME ≤ 8`. The record at `4` was then never written: `PRED` and `IPRED` came
//! back `NaN`, and so did the objective.
//!
//! `tied_obs_split.csv` has one subject per tie shape, a bolus of 100 at `TIME 0`, the tie at
//! `TIME 4` and one more observation at `TIME 8`:
//!
//! | ID | tie at `TIME 4`, in file order | split | before the fix |
//! |---|---|---|---|
//! | 1 | obs, `EVID=2`, obs | yes | second obs `NaN` |
//! | 2 | obs, obs, `EVID=2` | yes | second obs `NaN` |
//! | 3 | obs, obs | no | correct |
//! | 4 | obs, dose 50 (`ALAG1`), obs | yes | second obs `NaN` |
//! | 5 | obs, obs, dose 50 (`ALAG1`) | no (both move) | correct |
//!
//! NONMEM 7.6.0 ran `tied_obs_split.ctl` (`ADVAN1 TRANS2`, `ALAG1`, `METHOD=1 MAXEVAL=0
//! POSTHOC`); the ferx twin is `tied_obs_split_fit.ferx`, an `[odes]` model, since only the ODE
//! engines carried the defect. `maxiter = 0` removes the optimizer from both sides, so the
//! objective, the EBEs and every prediction are compared at the initial estimates.
//!
//! Non-degeneracy: subjects 3 and 5 are the controls, whose tie the reader leaves as one
//! instant, so the split is asserted on both sides of its gate before anything is compared.
//! The covariate `X` is constant, so these subjects take the dense engines, not the
//! event-driven one.
//!
//! Tier 2: one `fit()` at `maxiter = 0`.

use ferx_core::{fit, parse_full_model_file, read_nonmem_csv};
use std::path::Path;

/// NONMEM OBJ, the `-1000000000` row of `nonmem_anchor/results/tied_obs_split.ext`.
const NONMEM_OBJ: f64 = 2.672_681_311_519_760_1;

/// `(ID, TIME, PRED, IPRED, ETA1)` of every observation in
/// `nonmem_anchor/results/tied_obs_split.tab`.
fn nonmem_rows() -> Vec<(String, f64, f64, f64, f64)> {
    let path = "nonmem_anchor/results/tied_obs_split.tab";
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text
        .lines()
        .skip_while(|l| !l.trim_start().starts_with("ID"));
    let header: Vec<&str> = lines
        .next()
        .expect(".tab header")
        .split_whitespace()
        .collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let (id, time, evid, pred, ipred, eta) = (
        col("ID"),
        col("TIME"),
        col("EVID"),
        col("PRED"),
        col("IPRED"),
        col("ETA1"),
    );
    lines
        .map(|l| {
            l.split_whitespace()
                .map(|v| {
                    v.parse::<f64>()
                        .unwrap_or_else(|e| panic!("{path}: {v}: {e}"))
                })
                .collect::<Vec<f64>>()
        })
        .filter(|r| r[evid] == 0.0)
        .map(|r| {
            (
                format!("{}", r[id] as u32),
                r[time],
                r[pred],
                r[ipred],
                r[eta],
            )
        })
        .collect()
}

#[test]
fn a_reader_split_tie_matches_nonmem() {
    let parsed = parse_full_model_file(Path::new("nonmem_anchor/tied_obs_split_fit.ferx"))
        .unwrap_or_else(|e| panic!("{e}"));
    let pop = read_nonmem_csv(Path::new("nonmem_anchor/tied_obs_split.csv"), None, None)
        .unwrap_or_else(|e| panic!("{e}"));

    // The fixture reaches the split, and only where the table above says: without it the
    // anchor would pass on the bit-identical pair every engine always handled.
    let four = 4.0_f64;
    for (id, split) in [
        ("1", true),
        ("2", true),
        ("3", false),
        ("4", true),
        ("5", false),
    ] {
        let s = pop.subjects.iter().find(|s| s.id == id).expect("subject");
        let tie: Vec<f64> = s
            .obs_times
            .iter()
            .copied()
            .filter(|t| (t - four).abs() < 1e-9)
            .collect();
        assert_eq!(tie.len(), 2, "subject {id}: the tie holds two observations");
        assert_eq!(
            tie[0].to_bits() != tie[1].to_bits(),
            split,
            "subject {id}: tie read as {tie:?}, expected split = {split}"
        );
        if split {
            assert_eq!(
                (tie[0].to_bits(), tie[1].to_bits()),
                (four.next_down().to_bits(), four.to_bits()),
                "subject {id}: the split is one ULP below the tie"
            );
        }
    }

    let result = fit(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        &parsed.fit_options,
    )
    .unwrap_or_else(|e| panic!("fit: {e}"));

    // Finite first: a NaN fails every `<` below, but say so by name.
    assert!(result.ofv.is_finite(), "objective {}", result.ofv);
    let ofv_err = (result.ofv - NONMEM_OBJ).abs();
    assert!(
        ofv_err < OFV_TOL,
        "ferx OFV {:.12} vs NONMEM {NONMEM_OBJ:.12} (|error| {ofv_err:.3e})",
        result.ofv
    );

    let nm = nonmem_rows();
    let ferx: Vec<(&str, f64, f64, f64)> = result
        .subjects
        .iter()
        .flat_map(|s| {
            let sub = pop.subjects.iter().find(|p| p.id == s.id).expect("subject");
            (0..s.pred.len()).map(move |j| (s.id.as_str(), sub.obs_times[j], s.pred[j], s.ipred[j]))
        })
        .collect();
    assert_eq!(ferx.len(), nm.len(), "observation count");
    let (mut worst_pred, mut worst_ipred, mut worst_eta) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&(id, t, pred, ipred), (nm_id, nm_t, nm_pred, nm_ipred, nm_eta)) in ferx.iter().zip(&nm) {
        assert!(
            id == nm_id && (t - nm_t).abs() < 1e-9,
            "ferx row ({id}, {t}) is NONMEM's ({nm_id}, {nm_t})"
        );
        assert!(
            pred.is_finite() && ipred.is_finite(),
            "ID {id} TIME {t}: PRED {pred}, IPRED {ipred}"
        );
        let eta = result.subjects.iter().find(|s| s.id == id).unwrap().eta[0];
        let (e_pred, e_ipred, e_eta) = (
            (pred - nm_pred).abs(),
            (ipred - nm_ipred).abs(),
            (eta - nm_eta).abs(),
        );
        assert!(
            e_pred < PRED_TOL && e_ipred < IPRED_TOL && e_eta < ETA_TOL,
            "ID {id} TIME {t}: ferx PRED {pred} / IPRED {ipred} / η {eta} vs NONMEM \
             {nm_pred} / {nm_ipred} / {nm_eta}"
        );
        worst_pred = worst_pred.max(e_pred);
        worst_ipred = worst_ipred.max(e_ipred);
        worst_eta = worst_eta.max(e_eta);
    }
    println!(
        "tied_obs_split: |OFV − NONMEM| {ofv_err:.3e}; worst |PRED − NONMEM| {worst_pred:.3e}, \
         |IPRED − NONMEM| {worst_ipred:.3e}, |η − NONMEM| {worst_eta:.3e} over {} observations",
        nm.len()
    );
}

// Measured with the fix on Linux aarch64 and on macOS arm64, the same in every printed digit,
// against NONMEM 7.6.0: |OFV − NONMEM| 1.13e-8, worst |PRED − NONMEM| 9.7e-11
// (the `.tab` prints 11 significant figures, about 1e-10 at these magnitudes), |η − NONMEM|
// 3.98e-8 (where the two inner optimizers stop) and |IPRED − NONMEM| 1.89e-7, which is that η
// error times ∂IPRED/∂η ≈ 4.7. Each bound is 10× or more above its measured error. Before the
// fix the objective and three PREDs were `NaN`, so every bound here is far tighter than the
// defect.
const OFV_TOL: f64 = 1e-6;
const PRED_TOL: f64 = 1e-8;
const IPRED_TOL: f64 = 2e-6;
const ETA_TOL: f64 = 5e-7;
