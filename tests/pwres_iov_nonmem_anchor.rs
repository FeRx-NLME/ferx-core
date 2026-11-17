//! NONMEM `$TABLE EPRED EWRES` anchor for the Monte-Carlo population-weighted
//! residual (#2871).
//!
//! `compute_npde_npd` returns, beside NPDE/NPD, the replicate mean (`EPRED`) and
//! `Σ_sim^(-1/2) (y − EPRED)` (`PWRES`, NONMEM's `EWRES`) from the same simulation.
//! This checks both against NONMEM on the #734 IOV anchor: the same model, data
//! and `FIX`ed parameters as `npde_iov_nonmem_anchor.rs`
//! (`tests/fixtures/npde_iov_anchor.ferx` ↔ `nonmem_anchor/pwres_iov_*.ctl`), so
//! the two tools build the reference distribution from one population model and
//! differ only in their Monte-Carlo stream (`ESAMPLE=2000 SEED=2871` ↔
//! `nsim = 2000, seed = 2871`). Nine correlated observations per subject (three
//! occasions linked by the between-subject η), so the decorrelation is far from
//! the identity.
//!
//! # Two NONMEM tables, because the decorrelation is a convention
//!
//! | table | `$TABLE` | ferx EPRED | ferx PWRES |
//! |---|---|---|---|
//! | `pwres_iov_anchor.tab` | default (symmetric root) | matches | **matches** |
//! | `pwres_iov_chol.tab` | `WRESCHOL` (Cholesky) | matches, bit-identical EPRED | **differs** |
//!
//! ferx's PWRES follows NONMEM's default — the opposite of its NPDE, which follows
//! the Brendel/Comets Cholesky convention and so matches the `WRESCHOL` NPDE table
//! (`npde_iov_nonmem_anchor.rs`). The two choices are independent; this file pins
//! the PWRES one in both directions.
//!
//! # Measured tolerances
//!
//! Both sides are deterministic (fixed seeds, committed tables), so these are exact
//! realised numbers (540 observation rows):
//!
//! - EPRED: **worst relative |Δ| 0.02671** — two independent 2000-draw means of a
//!   quantity whose CV is ≈ 0.3–0.45, so `se ≈ 0.01` relative per side.
//! - EWRES vs the default table: **worst |Δ| 0.20688, rms 0.05251**.
//! - EWRES vs the `WRESCHOL` table: worst |Δ| 1.80922, **rms 0.50449** — the
//!   control, 9.6× the default table's rms.
//!
//! The bounds are 2× the realised values. The control is asserted on the rms,
//! which the 540 rows average, rather than on the worst row, which a single noisy
//! subject sets: it must clear the rms bound by 3×.
//!
//! # What this anchor cannot see
//!
//! A mutation reporting `PRED` (η = 0) as `EPRED` passes here: on this design the
//! expected and typical-value concentrations differ by less than the EPRED bound.
//! `epred_and_pwres_match_the_closed_form_moments_end_to_end`
//! (`src/stats/npde_pwres_tests.rs`) is the test that dies on it, on a model whose
//! `E[y]` sits 13% above `PRED` in closed form. The decorrelation mutations
//! (Cholesky for the symmetric root, diagonal-only) die here as well as there.

use ferx_core::stats::npde::{compute_npde_npd, SubjectNpde};
use ferx_core::types::{CompiledModel, Population};
use ferx_core::{parse_model_file, read_nonmem_csv};
use std::path::Path;

const MODEL: &str = "tests/fixtures/npde_iov_anchor.ferx";
const DATA: &str = "nonmem_anchor/npde_iov_anchor.csv";
/// Mirrors `ESAMPLE` / `SEED` on the `.ctl`s' `$TABLE`.
const NSIM: usize = 2000;
const SEED: u64 = 2871;

/// Realised worst relative |ΔEPRED| 0.02671, worst |ΔEWRES| 0.20688, rms ΔEWRES
/// 0.05251 against the default table; 2× headroom.
const EPRED_REL_TOL: f64 = 0.054;
const PWRES_TOL: f64 = 0.42;
const PWRES_RMS_TOL: f64 = 0.105;

/// `(ID, TIME, EPRED, EWRES)` per observation record of a committed `$TABLE`.
/// Dose records carry `DV = 0` and the design places no observation at a dose
/// time, so they are dropped by `DV > 0`.
fn nonmem_rows(file: &str) -> Vec<(String, f64, f64, f64)> {
    let text = std::fs::read_to_string(Path::new("nonmem_anchor/results").join(file))
        .unwrap_or_else(|e| panic!("the committed NONMEM table must be readable: {e}"));
    let mut lines = text.lines();
    lines.next().expect("TABLE NO. line");
    let header: Vec<&str> = lines
        .next()
        .expect("column header line")
        .split_whitespace()
        .collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("the table must carry a `{name}` column: {header:?}"))
    };
    let (id, time, dv, epred, ewres) = (
        col("ID"),
        col("TIME"),
        col("DV"),
        col("EPRED"),
        col("EWRES"),
    );
    lines
        .filter_map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .map(|v| v.parse().expect("a table cell is a number"))
                .collect();
            (f[dv] > 0.0).then(|| (format!("{}", f[id] as i64), f[time], f[epred], f[ewres]))
        })
        .collect()
}

fn anchor_scores() -> (Vec<SubjectNpde>, Population) {
    let model: CompiledModel = parse_model_file(Path::new(MODEL)).expect("the anchor model parses");
    let pop = read_nonmem_csv(Path::new(DATA), None, Some("OCC")).expect("the anchor data loads");
    let out = compute_npde_npd(&model, &pop, &model.default_params, NSIM, Some(SEED), None)
        .expect("npde");
    (out, pop)
}

/// `(worst relative |ΔEPRED|, worst |ΔPWRES|, rms ΔPWRES)` against a table.
///
/// Every operand is asserted finite before it is folded: `f64::max` discards a
/// `NaN`, so a PWRES that came back `NaN` would otherwise pass on the rows that
/// worked.
fn gaps(ferx: &[SubjectNpde], pop: &Population, table: &str) -> (f64, f64, f64) {
    let nm = nonmem_rows(table);
    let (mut worst_epred, mut worst_pwres, mut ss) = (0.0f64, 0.0f64, 0.0f64);
    let mut matched = 0usize;
    for (subj, s) in pop.subjects.iter().zip(ferx) {
        for (j, t) in subj.obs_times.iter().enumerate() {
            let &(_, _, want_epred, want_ewres) = nm
                .iter()
                .find(|(id, time, _, _)| *id == subj.id && (*time - t).abs() < 1e-9)
                .unwrap_or_else(|| panic!("no NONMEM row for subject {} at t = {t}", subj.id));
            assert!(
                s.epred[j].is_finite() && s.pwres[j].is_finite(),
                "subject {} t = {t}: non-finite ferx EPRED {} / PWRES {}",
                subj.id,
                s.epred[j],
                s.pwres[j]
            );
            assert!(want_epred.is_finite() && want_ewres.is_finite());
            let rel = (s.epred[j] / want_epred - 1.0).abs();
            let d = s.pwres[j] - want_ewres;
            assert!(rel.is_finite() && d.is_finite());
            worst_epred = worst_epred.max(rel);
            worst_pwres = worst_pwres.max(d.abs());
            ss += d * d;
            matched += 1;
        }
    }
    assert_eq!(
        matched,
        nm.len(),
        "every NONMEM observation row must be matched"
    );
    assert_eq!(matched, 540, "the anchor design is 60 x 3 x 3 observations");
    (worst_epred, worst_pwres, (ss / matched as f64).sqrt())
}

/// ferx's EPRED and PWRES are NONMEM's EPRED and default-convention EWRES, row by
/// row, to the Monte-Carlo error of two independent 2000-draw references.
#[test]
fn epred_and_pwres_match_nonmem_ewres() {
    let (out, pop) = anchor_scores();
    let (epred, pwres, rms) = gaps(&out, &pop, "pwres_iov_anchor.tab");
    eprintln!(
        "#2871 vs NONMEM (default): worst rel |dEPRED| = {epred:.5}, \
         worst |dEWRES| = {pwres:.5}, rms {rms:.5}"
    );
    assert!(
        epred < EPRED_REL_TOL,
        "worst relative |dEPRED| = {epred:.5} exceeds {EPRED_REL_TOL} (realised 0.02671)"
    );
    assert!(
        pwres < PWRES_TOL,
        "worst |dEWRES| = {pwres:.5} exceeds {PWRES_TOL} (realised 0.20688): ferx PWRES \
         is not NONMEM's default EWRES"
    );
    assert!(
        rms < PWRES_RMS_TOL,
        "rms dEWRES = {rms:.5} exceeds {PWRES_RMS_TOL} (realised 0.05251)"
    );
}

/// **The straddle: the anchor must be able to fail.** Against the `WRESCHOL`
/// table — the same simulation (bit-identical EPRED), decorrelated with the
/// Cholesky factor instead — PWRES must *not* agree. If it did, the default-table
/// match above would not be pinning the symmetric convention.
#[test]
fn pwres_disagrees_with_nonmem_wreschol_ewres() {
    let (out, pop) = anchor_scores();
    let (epred_chol, pwres_chol, rms) = gaps(&out, &pop, "pwres_iov_chol.tab");
    let (epred_default, _, _) = gaps(&out, &pop, "pwres_iov_anchor.tab");
    eprintln!("#2871 vs NONMEM (WRESCHOL): worst |dEWRES| = {pwres_chol:.5}, rms {rms:.5}");
    assert!(
        (epred_chol - epred_default).abs() < 1e-12,
        "EPRED must be identical between the two NONMEM tables; WRESCHOL changes only \
         the decorrelation"
    );
    // Realised rms 0.50449, 4.8× the rms bound the default table passes at.
    assert!(
        rms > 3.0 * PWRES_RMS_TOL,
        "ferx PWRES agrees with NONMEM's WRESCHOL EWRES to rms {rms:.5} (worst \
         {pwres_chol:.5}) — expected a visible disagreement (realised rms 0.50449): \
         ferx's PWRES follows NONMEM's default symmetric-root convention, not the \
         Cholesky one"
    );
}
