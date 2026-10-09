//! NONMEM `$TABLE NPDE NPD` anchor for a `block_sigma` reference distribution (#1733).
//!
//! `compute_npde_npd` drew every simulated residual from the model's **declared**
//! `block_sigma` correlation, one row at a time. A fit that estimated ρ was scored
//! at its starting value, and a cross-endpoint pair was scored as if its two
//! residuals were independent. The fix draws at the parameter set's ρ, through
//! the dense-`R` draw `simulate()` uses. This anchor checks that the result is the
//! reference distribution NONMEM builds at the same ρ.
//!
//! # The anchors
//!
//! Each ferx twin (`tests/fixtures/npde_block_sigma_{a,b}.ferx`) **declares**
//! ρ = 0.5 with a free `block_sigma` and is scored at a parameter set carrying
//! the **fitted** ρ. The NONMEM stream (`nonmem_anchor/npde_block_sigma_{a,b}.ctl`)
//! FIXes that fitted ρ in `$SIGMA BLOCK(2)`. Everything else is FIXed and identical,
//! so the two tools differ only in their Monte-Carlo stream
//! (`ESAMPLE=2000 SEED=1733` ↔ `nsim = 2000, seed = 1733`). `WRESCHOL` selects the
//! Cholesky decorrelation ferx uses (see `npde_iov_nonmem_anchor.rs`).
//!
//! * **Arm A**: `combined(PROP, ADD)`, σ = (0.2, 1.0), fitted ρ = −0.9 within one
//!   observation, i.e. the marginal variance. 30 subjects, two doses (the second
//!   on residual drug), seven observations each, `F` spanning about 1–20 so the
//!   cross term `2ρ·σ₁σ₂·F` is live on every row.
//! * **Arm B**: a total (`FREE = 0`) / unbound (`FREE = 1`) pair per sample, sharing
//!   an `L2` id, proportional σ = (0.05, 0.30), fitted ρ = −0.8 **across** the pair.
//!   30 subjects × four pairs. Whether NONMEM's `$TABLE NPDE` simulation honours
//!   `L2` was unknown when #1733 was planned. Measured: it does. Its NPDE matches
//!   ferx's joint draw at the fitted ρ and disagrees with an independent one.
//!
//! # Measured
//!
//! Linux x86_64 (`slow-tests.yml` run 37947001684, `nocapture`, at `42ec2252`) and
//! macOS arm64 (debug) agree with each other on every printed digit below.
//!
//! | arm | ρ ferx draws at | worst \|ΔNPD\| | worst \|ΔNPDE\| |
//! |---|---|---|---|
//! | A | fitted −0.9 | **0.43894** | **0.28557** |
//! | A | declared 0.5 (pre-#1733) | 1.69407 | 1.75498 |
//! | A | 0 | 1.43112 | 1.63312 |
//! | B | fitted −0.8 | **0.11838** (`FREE = 0` rows) | **0.28835** |
//! | B | declared 0.5 (pre-#1733) | — | 2.55441 |
//! | B | 0 (independent rows, pre-#1733) | — | 1.73352 |
//!
//! The fitted-ρ numbers are the Monte-Carlo disagreement of two independent
//! 2000-draw empirical CDFs, largest in the tails: arm A's worst NPD is one record
//! at `pd ≈ 0.998`. The bounds are 2× those. Both pre-#1733 configurations are
//! controls that must fail the bounds, so the anchor can fail.
//!
//! **NPD on the second row of an `L2` record is a NONMEM convention, not a ferx
//! defect.** On `FREE = 1` rows NONMEM's NPD tracks the *partner* row's score
//! (subject 1 t = 24: partner −1.607, NONMEM −1.601, ferx's own marginal −1.341).
//! The disagreement is 2.13 at every ρ tried (−0.8, 0, 0.5), so it does not depend
//! on the quantity under test. ferx's NPD is the row's own marginal (Brendel/Comets),
//! and #1733 does not touch it. It is pinned below as a disagreement, so that
//! "make NPD match NONMEM" cannot land unnoticed.

use ferx_core::stats::npde::{compute_npde_npd, SubjectNpde};
use ferx_core::types::{CompiledModel, ModelParameters, Population, ResidualCorrelation};
use ferx_core::{parse_model_file, read_nonmem_csv};
use std::path::Path;

/// Mirrors `ESAMPLE` / `SEED` on both `.ctl`s' `$TABLE`.
const NSIM: usize = 2000;
const SEED: u64 = 1733;

/// Arm A: 2× the realised worst |ΔNPD| 0.43894 and |ΔNPDE| 0.28557.
const A_NPD_TOL: f64 = 0.88;
const A_NPDE_TOL: f64 = 0.57;
/// Arm B: 2× the realised worst |ΔNPDE| 0.28835 and |ΔNPD| 0.11838 (`FREE = 0` rows).
const B_NPDE_TOL: f64 = 0.58;
const B_NPD_TOL: f64 = 0.24;

/// `(ID, TIME, FREE, NPD, NPDE)` per observation record (`MDV = 0`) of a table.
fn nonmem_scores(file: &str) -> Vec<(String, f64, f64, f64, f64)> {
    let text = std::fs::read_to_string(Path::new("nonmem_anchor/results").join(file))
        .unwrap_or_else(|e| panic!("the committed NONMEM table must be readable: {e}"));
    let mut lines = text.lines();
    lines.next().expect("TABLE NO. line");
    let header: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
    let col = |name: &str| header.iter().position(|h| *h == name);
    let c = |name: &str| col(name).unwrap_or_else(|| panic!("no `{name}` column: {header:?}"));
    let (id, time, mdv, npde, npd) = (c("ID"), c("TIME"), c("MDV"), c("NPDE"), c("NPD"));
    let free = col("FREE");
    lines
        .filter_map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .map(|v| v.parse().expect("a table cell is a number"))
                .collect();
            // NONMEM writes ID as a float; ferx keys subjects by the CSV string.
            (f[mdv] == 0.0).then(|| {
                (
                    format!("{}", f[id] as i64),
                    f[time],
                    free.map_or(0.0, |k| f[k]),
                    f[npd],
                    f[npde],
                )
            })
        })
        .collect()
}

fn load(arm: &str) -> (CompiledModel, Population) {
    let model = parse_model_file(Path::new(&format!(
        "tests/fixtures/npde_block_sigma_{arm}.ferx"
    )))
    .expect("the anchor model parses");
    assert_eq!(model.residual_correlations.len(), 1);
    assert!(
        (model.residual_correlations[0].rho - 0.5).abs() < 1e-12,
        "the twin must declare rho = 0.5, away from the fitted value"
    );
    let pop = read_nonmem_csv(
        Path::new(&format!("nonmem_anchor/npde_block_sigma_{arm}.csv")),
        None,
        None,
    )
    .expect("the anchor data loads");
    (model, pop)
}

/// The twin's parameters with ρ replaced, as a fit that estimated it returns them.
fn at_rho(model: &CompiledModel, rho: f64) -> ModelParameters {
    let mut p = model.default_params.clone();
    p.residual_correlations = vec![ResidualCorrelation {
        rho,
        ..model.residual_correlations[0].clone()
    }];
    p
}

fn free_of(subject: &ferx_core::types::Subject, j: usize) -> f64 {
    subject
        .obs_covariates
        .get(j)
        .and_then(|m| m.get("FREE").copied())
        .unwrap_or(0.0)
}

#[derive(Debug)]
struct Worst {
    /// |ΔNPD| over `FREE = 0` rows (all rows in arm A).
    npd_free0: f64,
    /// |ΔNPD| over `FREE = 1` rows (none in arm A).
    npd_free1: f64,
    npde: f64,
}

/// Worst |Δ| against a committed NONMEM table. `is_finite` is asserted per row,
/// not folded: `f64::max` discards a `NaN`, so a reference distribution that came
/// back `NaN` would otherwise pass on the strength of the rows that worked.
fn worst_vs_nonmem(ferx: &[SubjectNpde], pop: &Population, table: &str, n_rows: usize) -> Worst {
    let nm = nonmem_scores(table);
    let mut w = Worst {
        npd_free0: 0.0,
        npd_free1: 0.0,
        npde: 0.0,
    };
    let mut matched = 0usize;
    for (subj, s) in pop.subjects.iter().zip(ferx) {
        for (j, t) in subj.obs_times.iter().enumerate() {
            let fr = free_of(subj, j);
            let row = nm
                .iter()
                .find(|(id, time, free, _, _)| {
                    *id == subj.id && (*time - t).abs() < 1e-9 && (*free - fr).abs() < 1e-9
                })
                .unwrap_or_else(|| panic!("no NONMEM row for {} t {t} FREE {fr}", subj.id));
            assert!(
                s.npd[j].is_finite() && s.npde[j].is_finite(),
                "subject {} t {t}: ferx returned a non-finite score (npd {}, npde {})",
                subj.id,
                s.npd[j],
                s.npde[j]
            );
            let d_npd = (s.npd[j] - row.3).abs();
            if fr == 1.0 {
                w.npd_free1 = w.npd_free1.max(d_npd);
            } else {
                w.npd_free0 = w.npd_free0.max(d_npd);
            }
            w.npde = w.npde.max((s.npde[j] - row.4).abs());
            matched += 1;
        }
    }
    assert_eq!(
        matched,
        nm.len(),
        "every NONMEM observation row must be matched"
    );
    assert_eq!(matched, n_rows, "the anchor design's observation count");
    w
}

fn run(arm: &str, rho: f64, n_rows: usize) -> Worst {
    let (model, pop) = load(arm);
    let out = compute_npde_npd(&model, &pop, &at_rho(&model, rho), NSIM, Some(SEED)).expect("npde");
    let w = worst_vs_nonmem(&out, &pop, &format!("npde_block_sigma_{arm}.tab"), n_rows);
    eprintln!("#1733 arm {arm} at rho {rho}: {w:?}");
    w
}

/// Arm A: the marginal variance at the fitted within-observation ρ.
#[test]
fn npde_at_the_fitted_within_observation_rho_matches_nonmem() {
    let w = run("a", -0.9, 210);
    assert!(
        w.npd_free0 < A_NPD_TOL,
        "worst |dNPD| = {:.5} exceeds {A_NPD_TOL} (realised 0.43894): the residual \
         draw is not at the fitted block_sigma rho",
        w.npd_free0
    );
    assert!(
        w.npde < A_NPDE_TOL,
        "worst |dNPDE| = {:.5} exceeds {A_NPDE_TOL} (realised 0.28557)",
        w.npde
    );
}

/// Arm A's straddle: drawing at the declared ρ (the pre-#1733 behaviour), or at
/// ρ = 0, must fail both bounds.
#[test]
fn npde_at_the_declared_rho_diverges_from_nonmem_arm_a() {
    for (rho, label) in [(0.5, "declared"), (0.0, "zero")] {
        let w = run("a", rho, 210);
        assert!(
            w.npd_free0 > A_NPD_TOL && w.npde > A_NPDE_TOL,
            "the {label}-rho control agrees with NONMEM to |dNPD| {:.5} / |dNPDE| {:.5}, \
             inside the anchor's own bounds ({A_NPD_TOL} / {A_NPDE_TOL}): the anchor no \
             longer straddles #1733 (realised 1.69 / 1.75 declared, 1.43 / 1.63 zero)",
            w.npd_free0,
            w.npde
        );
    }
}

/// Arm B: the cross-endpoint pair is drawn jointly at the fitted ρ. NPDE is the
/// decorrelated score, so it is the one that sees the covariance between rows.
#[test]
fn npde_of_an_l2_pair_at_the_fitted_rho_matches_nonmem() {
    let w = run("b", -0.8, 240);
    assert!(
        w.npde < B_NPDE_TOL,
        "worst |dNPDE| = {:.5} exceeds {B_NPDE_TOL} (realised 0.28835): the paired \
         residuals are not drawn jointly at the fitted rho",
        w.npde
    );
    assert!(
        w.npd_free0 < B_NPD_TOL,
        "worst |dNPD| on FREE = 0 rows = {:.5} exceeds {B_NPD_TOL} (realised 0.11838)",
        w.npd_free0
    );
    // NONMEM's NPD on the second row of an L2 record tracks the partner row; ferx's
    // is the row's own marginal. Realised 2.13499. See the module docs.
    assert!(
        w.npd_free1 > 1.0,
        "ferx NPD on FREE = 1 rows agrees with NONMEM's to {:.5}: expected the \
         second-row-of-L2 convention gap (realised 2.13)",
        w.npd_free1
    );
}

/// Arm B's straddle: an independent per-row draw (ρ = 0, the pre-#1733 draw
/// for any ρ) and the declared ρ must both fail the NPDE bound.
#[test]
fn npde_of_an_l2_pair_drawn_independently_diverges_from_nonmem() {
    for (rho, label) in [(0.5, "declared"), (0.0, "independent")] {
        let w = run("b", rho, 240);
        assert!(
            w.npde > B_NPDE_TOL,
            "the {label} control agrees with NONMEM to |dNPDE| {:.5}, inside \
             {B_NPDE_TOL}: the anchor no longer straddles #1733 (realised 2.55 \
             declared, 1.73 independent)",
            w.npde
        );
    }
}
