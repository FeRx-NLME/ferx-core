//! NONMEM anchor for a **`FIX`-ed** `block_sigma` correlation past ferx's
//! Fisher-z estimation rail (#1307).
//!
//! `src/types.rs` documents `block_sigma (...) = [...] FIX` as *"holds it at the
//! declared value"*. It did not, above `|ρ| ≈ 0.995`: `pack_rho` clamped into
//! `±RHO_Z_BOUND` before any bound was consulted, so every declared ρ past
//! `tanh(3) = 0.995_055` collapsed onto that one number and the fit scored a
//! covariance the model file never wrote.
//!
//! # The anchor
//!
//! `nonmem_anchor/correlated_residual_rho999.ctl` is `$SIGMA BLOCK(2) FIX` with
//! every THETA and the OMEGA fixed too, so the objective is a pure function of
//! the declared Σ — there is nothing left for an optimizer to absorb a
//! substituted ρ into. `nonmem_anchor/correlated_residual_rho999_fit.ferx` is
//! the ferx twin, on the same `data/correlated_residual_combined.csv`.
//!
//! | | OFV |
//! |---|---|
//! | NONMEM 7.5.1, `METHOD=1 MAXEVAL=0` | **442.272 447 308 448 70** |
//! | ferx, after #1307 | **442.138 163** — Δ = 0.134 |
//! | ferx, before #1307 (ρ substituted by −0.995055) | **301.450 784 063 187 03** — Δ = 140.821 663 |
//!
//! # Why the block is *balanced*, which is the whole design
//!
//! The first attempt at this anchor used the sibling fixture's `σ_prop = 0.2`,
//! `σ_add = 0.3` at ρ = +0.999. Measured: NONMEM 14.155997, ferx 14.175702 after
//! the fix and 14.1636 before it. The **pre-fix** value was the closer of the
//! two. The cross term `2·IPRED·ρ·σ₁·σ₂` is a twentieth of the diagonal there,
//! so the entire ρ = 0.995 → 0.999 effect is 0.012 OFV, while this fixture's own
//! FOCE-vs-NONMEM baseline gap is 0.005–0.02 — an oracle measuring itself, in
//! exactly the way AGENTS.md's "the oracle has to be more accurate than the
//! difference you are about to call a defect" warns about.
//!
//! The fix is to make ρ bite. With `σ_prop = 0.03`, `σ_prop·IPRED` (0.12–0.30
//! over this dataset) is the same size as `σ_add = 0.3`, so
//! `IPRED²σ₁² + σ₂² + 2·IPRED·ρσ₁σ₂` nearly **cancels** as ρ → −1 and the
//! residual variance at the declared ρ is ~5× smaller than at the clamped one.
//! The separation is 1050× the residual disagreement, which is what makes the
//! comparison an oracle rather than a coin flip. The block stays
//! positive-definite (det = 1.62e-7 > 0), so NONMEM accepts it and ferx's `R`
//! is invertible — near-singular by the user's own declaration, which is
//! precisely what `FIX` is for.
//!
//! # Tier and gate
//!
//! Tier 2: `outer_maxiter = 0` is one evaluation and no convergence loop, so
//! this returns immediately and needs no `slow-tests` gate.

use ferx_core::{fit, read_nonmem_csv, run_model_with_data, FitOptions};
use std::path::Path;

/// NONMEM 7.5.1, `$EST METHOD=1 MAXEVAL=0 NOABORT`, read from the `.ext`
/// `-1000000000` row of `nonmem_anchor/correlated_residual_rho999.ctl`.
const NONMEM_OFV: f64 = 442.272_447_308_448_70;

/// ferx before #1307, with `pack_rho`'s `RHO_Z_BOUND` clamp applied to the
/// `FIX`-ed coordinate — i.e. scoring ρ = −0.995055 instead of the declared
/// −0.999. Not a target; it is here so the tolerance below can be *chosen*
/// against the failure it has to catch rather than picked.
const PRE_FIX_OFV: f64 = 301.450_784_063_187_03;

/// Realised |ferx − NONMEM| is **0.134** (0.03%), which is this fixture's
/// ordinary FOCE inner-EBE disagreement — the ρ = 0.5 sibling
/// (`correlated_residual_combined.ctl`) sits at 0.005 on an OFV of 18.7, the
/// same order relative to the objective's scale.
///
/// The bound is 0.5: ~3.7× the realised error, and **280×** below the 140.8 the
/// defect produced. Anywhere in `(0.14, 140)` discriminates, so this is not a
/// number the test is sensitive to — which is the point of quoting both
/// distances rather than one.
const TOL: f64 = 0.5;

fn ferx_ofv() -> f64 {
    let model = ferx_core::parse_model_file(Path::new(
        "nonmem_anchor/correlated_residual_rho999_fit.ferx",
    ))
    .expect("the anchor twin must parse");
    let population = read_nonmem_csv(
        Path::new("data/correlated_residual_combined.csv"),
        None,
        None,
    )
    .expect("the anchor dataset must load");

    // The model file's own `[fit_options]` are **not** carried by the parse, so
    // `maxiter = 0` is restated here on the `FitOptions` actually passed to
    // `fit()`. A model string that sets `method` there silently runs the
    // default otherwise.
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        method: ferx_core::EstimationMethod::Foce,
        ..FitOptions::default()
    };
    let result = fit(&model, &population, &model.default_params, &options)
        .expect("the anchor evaluation must succeed");

    // The premise, asserted rather than trusted: the declared ρ is what reached
    // the fit. Without this the OFV comparison could agree for some other
    // reason and the test would stop being about #1307.
    let corr = result.residual_correlations[0];
    assert!(
        (corr.rho + 0.999).abs() < 1e-9,
        "the anchor must score the declared ρ = −0.999; got {}",
        corr.rho
    );
    assert!(
        result.ofv.is_finite(),
        "a diverged solve returns a finite sentinel elsewhere in this crate, so \
         finiteness is asserted before any distance is taken"
    );
    result.ofv
}

/// #1307. ferx reproduces NONMEM's objective for a `FIX`-ed `$SIGMA BLOCK(2)`
/// at ρ = −0.999 — a correlation past ferx's own estimation rail, which is the
/// case `pack_rho` used to rewrite.
#[test]
fn a_fixed_sigma_block_past_the_rail_matches_nonmem() {
    let ofv = ferx_ofv();
    let err = (ofv - NONMEM_OFV).abs();
    assert!(
        err < TOL,
        "ferx OFV {ofv} vs NONMEM {NONMEM_OFV} — |Δ| = {err}, bound {TOL}"
    );
}

/// The same run, asserted against the **defect** rather than against NONMEM:
/// the pre-#1307 objective is 140 OFV away, so this arm fails loudly if the
/// clamp ever comes back — including in a future where the anchor's NONMEM
/// constant has been re-measured and the arm above re-tuned around it.
///
/// Two assertions, not one, because they can fail for different reasons: the
/// first is "we are not scoring the clamped ρ", the second is "the two
/// hypotheses are actually distinguishable at this tolerance", which is what
/// makes the first one mean something.
#[test]
fn the_pre_fix_objective_is_far_outside_the_anchor_tolerance() {
    let ofv = ferx_ofv();
    assert!(
        (ofv - PRE_FIX_OFV).abs() > 100.0,
        "ferx OFV {ofv} is within 100 of the pre-#1307 value {PRE_FIX_OFV}, i.e. \
         the clamped ρ is being scored again"
    );
    assert!(
        (PRE_FIX_OFV - NONMEM_OFV).abs() > 20.0 * TOL,
        "the fixture no longer separates the two hypotheses — a green anchor \
         above would say nothing"
    );
}

/// The whole-file path the CLI takes, so the anchor covers the model file as
/// shipped rather than only the `FitOptions` this test assembles. Its
/// `[fit_options]` block carries `method = foce` and `maxiter = 0`.
#[test]
fn the_shipped_anchor_model_file_reaches_the_same_objective() {
    let (result, _population) = run_model_with_data(
        "nonmem_anchor/correlated_residual_rho999_fit.ferx",
        Some("data/correlated_residual_combined.csv"),
    )
    .expect("the shipped anchor model must run end to end");
    let err = (result.ofv - NONMEM_OFV).abs();
    assert!(
        err < TOL,
        "ferx OFV {} vs NONMEM {NONMEM_OFV} — |Δ| = {err}, bound {TOL}",
        result.ofv
    );
}
