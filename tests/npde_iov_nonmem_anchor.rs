//! NONMEM `$TABLE NPDE NPD` anchor for the IOV reference distribution (#734).
//!
//! `compute_npde_npd` built its Monte-Carlo reference with every occasion κ held at
//! zero, so for an IOV model the reference carried no inter-occasion variability and
//! the scores came out over-dispersed — the diagnostic understating its own spread.
//! This is the cross-tool check that the fixed reference is the *same distribution*
//! NONMEM builds, not merely a wider one.
//!
//! # The anchor
//!
//! `nonmem_anchor/npde_iov_anchor.ctl` is a bit-for-bit twin of
//! `tests/fixtures/npde_iov_anchor.ferx` on the shared dataset
//! `npde_iov_anchor.csv` (60 subjects, three `EVID=4` reset+bolus occasions each,
//! three post-dose observations per occasion; DV simulated once under the same
//! θ/Ω/Ω_IOV/Σ — see `tests/gen_npde_iov_anchor.rs`). Every parameter is `FIX`ed at
//! the simulating value on both sides, so the two tools build the reference
//! distribution from the *same* population model and differ only in their
//! Monte-Carlo stream (`ESAMPLE=2000 SEED=734` ↔ `nsim = 2000, seed = 734`).
//!
//! # Two NONMEM tables, because NPDE's decorrelation is a convention
//!
//! Decorrelating a vector is not unique, and the two tools do not default to the
//! same factor. NONMEM's default NPDE uses the **symmetric** square root of the
//! simulated covariance (its CWRES convention); ferx follows Brendel/Comets and the
//! `npde` R package, which use the **Cholesky** factor — which NONMEM produces on
//! request via `WRESCHOL`. Both tables are committed and both are asserted here:
//!
//! | table | `$TABLE` | ferx NPD | ferx NPDE |
//! |---|---|---|---|
//! | `npde_iov_anchor.tab` | default | matches (corr 0.99895) | **differs** (corr 0.8682) |
//! | `npde_iov_chol.tab` | `WRESCHOL` | matches, bit-identical NPD | matches (corr 0.99723) |
//!
//! NPD carries no decorrelation, so it is bit-identical between the two NONMEM runs
//! — which is what makes the pair isolate the convention and nothing else. Pinning
//! the *disagreement* with the default table matters as much as the agreement:
//! without it, "make NPDE match NONMEM" is an invitation to adopt the wrong factor.
//!
//! # Measured tolerances
//!
//! Both sides are deterministic (fixed seeds, committed table), so these are exact
//! realised numbers, not estimates:
//!
//! - NPD:  mean Δ −0.00587, sd 0.04820, **worst |Δ| 0.22578** (ID 34, t = 49)
//! - NPDE: mean Δ  0.00063, sd 0.07591, **worst |Δ| 0.28044** (vs `WRESCHOL`)
//!
//! That is the Monte-Carlo disagreement of two independent 2000-draw empirical CDFs
//! (`se(pd) ≈ 0.011`, inflating to ≈0.05–0.15 on the score near the tails), not a
//! model difference. The bounds below are 2× the worst realised value. The
//! pre-#734 behaviour is reproduced by the `zero_omega_iov` control, which fails
//! both bounds — worst |ΔNPD| 0.89495 (2.0× the NPD bound) and worst |ΔNPDE|
//! 4.38933 (7.8× the NPDE bound) — so the anchor can fail.

use ferx_core::stats::npde::{compute_npde_npd, SubjectNpde};
use ferx_core::types::{CompiledModel, ModelParameters, Population};
use ferx_core::{parse_model_file, read_nonmem_csv};
use std::path::Path;

const MODEL: &str = "tests/fixtures/npde_iov_anchor.ferx";
const DATA: &str = "nonmem_anchor/npde_iov_anchor.csv";
/// Mirrors `ESAMPLE` / `SEED` on the `.ctl`'s `$TABLE`.
const NSIM: usize = 2000;
const SEED: u64 = 734;

/// Worst realised |Δ| against the `WRESCHOL` table: NPD 0.22578, NPDE 0.28044.
/// The bounds carry 2× headroom over those measurements.
const NPD_TOL: f64 = 0.45;
const NPDE_TOL: f64 = 0.56;

/// `(ID, TIME, NPD, NPDE)` per **observation** record of a committed `$TABLE`.
///
/// NONMEM tables dose records too; they carry `DV = 0` and zeroed scores, and the
/// design places no observation at a dose time, so they are dropped by `DV > 0`.
fn nonmem_scores(file: &str) -> Vec<(String, f64, f64, f64)> {
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
    let (id, time, dv, npde, npd) = (col("ID"), col("TIME"), col("DV"), col("NPDE"), col("NPD"));
    lines
        .filter_map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .map(|v| v.parse().expect("a table cell is a number"))
                .collect();
            // NONMEM writes ID as a float; ferx keys subjects by the CSV string.
            (f[dv] > 0.0).then(|| (format!("{}", f[id] as i64), f[time], f[npd], f[npde]))
        })
        .collect()
}

fn anchor_model_and_data() -> (CompiledModel, Population) {
    let model = parse_model_file(Path::new(MODEL)).expect("the anchor model parses");
    assert_eq!(
        model.n_kappa, 1,
        "the anchor must declare an occasion kappa"
    );
    let pop = read_nonmem_csv(Path::new(DATA), None, Some("OCC")).expect("the anchor data loads");
    assert!(
        pop.subjects.iter().all(|s| !s.occasions.is_empty()),
        "every subject must carry occasion labels, or there is no kappa to draw"
    );
    (model, pop)
}

/// Worst |Δ| of each score against a committed NONMEM table, plus the row that
/// produced the worst NPD.
///
/// `is_finite` is asserted per row rather than folded away: `f64::max` returns the
/// *other* operand for a `NaN`, so a reference distribution that came back `NaN` —
/// the likeliest way to break the thing under test — would leave the accumulators
/// at whatever the working rows produced and pass.
fn worst_vs_nonmem(
    ferx: &[SubjectNpde],
    pop: &Population,
    table: &str,
) -> (f64, f64, (String, f64)) {
    let nm = nonmem_scores(table);
    let mut worst_npd: f64 = 0.0;
    let mut worst_npde: f64 = 0.0;
    let mut worst_row = (String::new(), f64::NAN);
    let mut matched = 0usize;
    for (subj, s) in pop.subjects.iter().zip(ferx) {
        for (j, t) in subj.obs_times.iter().enumerate() {
            let (_, _, want_npd, want_npde) = nm
                .iter()
                .find(|(id, time, _, _)| *id == subj.id && (*time - t).abs() < 1e-9)
                .unwrap_or_else(|| panic!("no NONMEM row for subject {} at t = {t}", subj.id))
                .clone();
            assert!(
                s.npd[j].is_finite() && s.npde[j].is_finite(),
                "subject {} t = {t}: ferx returned a non-finite score (npd {}, npde {})",
                subj.id,
                s.npd[j],
                s.npde[j]
            );
            let d_npd = (s.npd[j] - want_npd).abs();
            if d_npd > worst_npd {
                worst_npd = d_npd;
                worst_row = (subj.id.clone(), *t);
            }
            worst_npde = worst_npde.max((s.npde[j] - want_npde).abs());
            matched += 1;
        }
    }
    assert_eq!(
        matched,
        nm.len(),
        "every NONMEM observation row must be matched"
    );
    assert_eq!(matched, 540, "the anchor design is 60 x 3 x 3 observations");
    (worst_npd, worst_npde, worst_row)
}

/// The reference distribution ferx builds must be NONMEM's, row by row.
#[test]
fn npde_iov_reference_matches_nonmem() {
    let (model, pop) = anchor_model_and_data();
    let out = compute_npde_npd(&model, &pop, &model.default_params, NSIM, Some(SEED));

    let (npd, npde, row) = worst_vs_nonmem(&out, &pop, "npde_iov_chol.tab");
    eprintln!(
        "#734 vs NONMEM (WRESCHOL): worst |dNPD| = {npd:.5} (ID {} t {}), \
         worst |dNPDE| = {npde:.5}",
        row.0, row.1
    );
    assert!(
        npd < NPD_TOL,
        "worst |dNPD| = {npd:.5} exceeds {NPD_TOL} (realised 0.22578): the ferx NPD \
         reference distribution is not NONMEM's — occasion kappa dropped, or drawn \
         at the wrong scale"
    );
    assert!(
        npde < NPDE_TOL,
        "worst |dNPDE| = {npde:.5} exceeds {NPDE_TOL} (realised 0.28044)"
    );

    // NPD carries no decorrelation, so it must match NONMEM's *default* table just
    // as well — the two NONMEM runs write bit-identical NPD columns.
    let (npd_default, npde_default, _) = worst_vs_nonmem(&out, &pop, "npde_iov_anchor.tab");
    assert!(
        (npd_default - npd).abs() < 1e-9,
        "NPD must be identical between the two NONMEM tables ({npd_default:.5} vs {npd:.5}); \
         WRESCHOL changes only the NPDE decorrelation"
    );
    // ...while NPDE must *not* match the default table: NONMEM decorrelates with
    // the symmetric square root there, ferx with the Cholesky factor. Realised
    // worst |dNPDE| 2.09409 (7.5× the WRESCHOL bound), corr 0.8682 against 0.99723.
    // Pinned so that "make NPDE match NONMEM's default" cannot land quietly — it
    // would be adopting the CWRES factor, not the Brendel/Comets one.
    assert!(
        npde_default > 2.0 * NPDE_TOL,
        "ferx NPDE agrees with NONMEM's DEFAULT decorrelation to {npde_default:.5} — \
         expected a visible disagreement (realised 2.09409), since ferx follows the \
         Cholesky (Brendel/Comets) convention and NONMEM defaults to the symmetric root"
    );
}

/// **The straddle: the anchor must be able to fail.**
///
/// Same seed, same everything, but `Ω_IOV` forced to zero — the pre-#734 reference,
/// reached with the RNG stream still aligned (the per-occasion κ draws happen and
/// scale to zero). If this arm also agreed with NONMEM, the test above would be
/// passing on something other than the occasion κ.
#[test]
fn npde_iov_zero_omega_iov_control_diverges_from_nonmem() {
    let (model, pop) = anchor_model_and_data();
    let mut zero_iov: ModelParameters = model.default_params.clone();
    {
        let om = zero_iov
            .omega_iov
            .as_mut()
            .expect("the anchor model carries an Ω_IOV");
        om.chol.fill(0.0);
        om.matrix.fill(0.0);
    }
    let out = compute_npde_npd(&model, &pop, &zero_iov, NSIM, Some(SEED));
    let (npd, npde, _) = worst_vs_nonmem(&out, &pop, "npde_iov_chol.tab");
    eprintln!("#734 zero-Ω_IOV control: worst |dNPD| = {npd:.5}, worst |dNPDE| = {npde:.5}");
    // Realised: 0.89495 and 4.38933, i.e. 2.0× and 7.8× the bounds the fixed path
    // passes at. Both are asserted — NPD alone clears its bound by only 2×, and it
    // is the decorrelated NPDE that shows the missing occasion component loudest.
    assert!(
        npd > NPD_TOL && npde > NPDE_TOL,
        "the zero-Ω_IOV control agrees with NONMEM to |dNPD| {npd:.5} / |dNPDE| \
         {npde:.5}, inside the anchor's own bounds ({NPD_TOL} / {NPDE_TOL}) — the \
         anchor no longer straddles the #734 fix and cannot fail"
    );
}
