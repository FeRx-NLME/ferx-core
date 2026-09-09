//! NONMEM anchor for additive (`+`) `[covariate_model]` relations (#1313).
//!
//! `CL = TVCL*EXP(ETA(1)) + THETA(6)*(WT - 70)` is an ordinary NONMEM model, so
//! this feature gets an ordinary anchored comparison — none of the CLAUDE.md
//! exceptions applies. Two arms, four control streams, all
//! `$EST MAXEVAL=0 POSTHOC INTERACTION` so nothing is estimated on either side
//! and the comparison is of arithmetic rather than of where two optimizers
//! stop:
//!
//! - **A, `additive_cov_lin.ctl`** — the additive combination on its own,
//!   `CL = TVCL*EXP(ETA(1)) + 0.04*(WT - 70)`.
//! - **B, `additive_cov_mixed.ctl`** — a multiplicative *and* an additive
//!   relation on the same `CL`. That is the placement rule the block gains
//!   along with the operator, and the readings differ numerically
//!   (`TVCL*f*EXP(ETA) + t` is neither `TVCL*f*(EXP(ETA) + t)` nor
//!   `(TVCL*EXP(ETA) + t)*f`), so NONMEM's own spelling settles it.
//! - **`*_null.ctl`** — each arm's twin with the additive slope held at `0 FIX`
//!   and nothing else changed. See "Why the objective is compared as a
//!   difference" below.
//!
//! Two objects are compared. `PRED` — the population prediction at `ETA = 0`,
//! where the covariate effects are the only thing acting — matches to `$TABLE`
//! print precision (`FORMAT=s1PE23.16`). The FOCEI objective is compared as the
//! **difference** each arm makes against its own null twin.
//!
//! # Why the objective is compared as a difference
//!
//! ferx and NONMEM do not report the same FOCEI objective on this dataset at
//! these initial estimates, and **that is true with no covariate model at all**:
//! the bare `CL = TVCL*exp(ETA_CL)` model reads `-1017.3100571` in ferx against
//! `-1033.3346138` in NONMEM, a 16.02 offset that #1313 neither introduces nor
//! touches (measured, not assumed — `additive_cov_lin_null.ctl` *is* that
//! model, and ferx reproduces its own value to 1e-9 across `inner_tol` 1e-10 ..
//! 1e-14 and `inner_restarts` 0 .. 8, so it is not an inner-EBE tolerance).
//! The offset is stable in the additive slope — ferx − NONMEM is 16.0246 at
//! slope 0 and 16.0247 at slope 0.04 — so the *difference* between an arm and
//! its null twin cancels it and isolates what the additive term contributes to
//! the objective, which is the quantity this feature owns. Chasing the baseline
//! offset itself belongs to the FOCEI-vs-NONMEM work (#864 and its successors),
//! not here.
//!
//! Non-degeneracy is in the data, not in a comment: `WT` runs 45.0 .. 93.7 about
//! a centre of 70 and `CRCL` runs 46.5 .. 150.0 about a centre of 100, so both
//! additive terms take **both signs** across the 30 subjects and move `CL` by up
//! to 25 % of `TVCL`. `the_additive_term_is_what_is_being_anchored` states that
//! as a test: with the slope zeroed the same comparison misses by 39 % (arm A)
//! and 43 % (arm B), thirteen orders outside the bound the anchors pass at.

use std::path::Path;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::FitOptions;
use ferx_core::{fit, predict, read_nonmem_csv};

/// ferx reads the observation records on `CMT=1` (`two_cpt_oral` puts central
/// there); the NONMEM streams run on `nonmem_anchor/covmodel_stall.csv`, which
/// is this file with observation `CMT` recoded 1 → 2 for `ADVAN4`. Nothing else
/// differs between the two datasets.
const DATA: &str = "data/two_cpt_oral_cov.csv";

/// Everything both arms share, at the initial estimates the control streams
/// declare. `method = focei` matches `METHOD=1 INTERACTION`.
const HEAD: &str = r"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(40.0, 1.0, 500.0)
  theta TVQ(8.0, 0.1, 100.0)
  theta TVV2(80.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)

  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  omega ETA_Q  ~ 0.08
  omega ETA_V2 ~ 0.08
  omega ETA_KA ~ 0.20

  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ  * exp(ETA_Q)
  V2 = TVV2 * exp(ETA_V2)
  KA = TVKA * exp(ETA_KA)

[covariates]
  WT   continuous
  CRCL continuous
";

const TAIL: &str = r"
[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
";

/// Arm A: `CL = TVCL * exp(ETA_CL) + THETA_CL_WT * (WT - 70)`.
fn arm_a(slope: f64) -> String {
    format!(
        "{HEAD}
[covariate_model]
  CL ~ WT linear(center = 70) + => THETA_CL_WT({slope}, -1.0, 1.0)
{TAIL}"
    )
}

/// Arm B: `CL = TVCL * (WT/70)^0.6 * exp(ETA_CL) + THETA_CL_CRCL * (CRCL - 100)`.
fn arm_b(slope: f64) -> String {
    format!(
        "{HEAD}
[covariate_model]
  CL ~ WT   power(center = 70)     => THETA_CL_WT(0.6, 0.01, 5.0)
  CL ~ CRCL linear(center = 100) + => THETA_CL_CRCL({slope}, -1.0, 1.0)
{TAIL}"
    )
}

/// `(ID, TIME, PRED)` per record from a committed `$TABLE`
/// (`FORMAT=s1PE23.16`), skipping the `TABLE NO.` and header lines.
///
/// The curated run outputs live under `nonmem_anchor/results/`; the top level
/// of `nonmem_anchor/` gitignores `*.tab` / `*.lst` as scratch.
fn nonmem_pred(file: &str) -> Vec<(String, f64, f64)> {
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
    let (id, time, pred) = (col("ID"), col("TIME"), col("PRED"));
    lines
        .map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .map(|v| v.parse().expect("a table cell is a number"))
                .collect();
            // NONMEM writes ID as a float; ferx keys subjects by the CSV string.
            (format!("{}", f[id] as i64), f[time], f[pred])
        })
        .collect()
}

/// The worst relative `PRED` difference between ferx and the NONMEM table over
/// the 300 observation records.
///
/// `is_finite` is asserted rather than folded away: `f64::max` returns the
/// *other* operand for a `NaN`, so a predictor returning `NaN` — the likeliest
/// way to break the thing under test — would leave the accumulator at whatever
/// the working records produced and pass.
fn worst_pred_error(model_src: &str, table: &str) -> f64 {
    let parsed = parse_full_model(model_src).expect("the ferx model must parse");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("the dataset must load");
    let ferx = predict(&parsed.model, &pop, &parsed.model.default_params);
    let nm = nonmem_pred(table);

    let mut worst: f64 = 0.0;
    let mut matched = 0usize;
    for row in &ferx {
        let Some((_, _, want)) = nm
            .iter()
            .find(|(id, time, _)| *id == row.id && (*time - row.time).abs() < 1e-9)
        else {
            panic!("no NONMEM row for subject {} at t = {}", row.id, row.time);
        };
        // The `TIME = 0` dose record tables `PRED = 0`; there is nothing to
        // compare there and a relative error would divide by zero.
        if *want == 0.0 {
            continue;
        }
        assert!(
            row.pred.is_finite(),
            "ferx PRED is not finite for subject {} at t = {}",
            row.id,
            row.time
        );
        worst = worst.max((row.pred - want).abs() / want.abs());
        matched += 1;
    }
    assert_eq!(
        matched, 300,
        "the anchor must compare all 300 observations, not a subset"
    );
    worst
}

// NONMEM `OBJECTIVE FUNCTION VALUE`, `$EST MAXEVAL=0 POSTHOC INTERACTION`.
const NM_OFV_A: f64 = -1037.4200935745573;
const NM_OFV_A_NULL: f64 = -1033.3346138081854;
const NM_OFV_B: f64 = -1036.7935175603372;
const NM_OFV_B_NULL: f64 = -1037.8010768417562;

/// The ferx FOCEI objective at the same parameter vector: an evaluation, not a
/// fit — `outer_maxiter = 0` is ferx's `MAXEVAL=0`.
fn ferx_ofv(model_src: &str) -> f64 {
    let parsed = parse_full_model(model_src).expect("the ferx model must parse");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("the dataset must load");
    let options = FitOptions {
        outer_maxiter: 0,
        ..parsed.fit_options.clone()
    };
    let result = fit(&parsed.model, &pop, &parsed.model.default_params, &options)
        .expect("the evaluation must run");
    assert!(result.ofv.is_finite(), "ferx OFV is not finite");
    result.ofv
}

#[test]
fn an_additive_relation_matches_nonmem() {
    // Measured, not guessed: the realised worst relative PRED error over all
    // 300 observations is 3.08e-15 — about 14 ulp, i.e. the `$TABLE` print
    // precision at `FORMAT=s1PE23.16`. The bound is 30× that, and still 13
    // orders below anything the combination rule could produce: with the term
    // dropped the same comparison reads 3.9e-1 (see the last test).
    let rel = worst_pred_error(&arm_a(0.04), "additive_cov_lin.tab");
    assert!(rel < 1e-13, "arm A PRED: worst relative error {rel:e}");

    // What the additive term contributes to the objective. Realised difference
    // between the two engines' deltas: 1.126e-4 on a delta of −4.0855 (2.8e-5
    // relative). The bound is 9× the measurement, and the baseline offset it
    // cancels is 16.02 — so an implementation that simply dropped the term
    // cannot pass it.
    let delta = ferx_ofv(&arm_a(0.04)) - ferx_ofv(&arm_a(0.0));
    let want = NM_OFV_A - NM_OFV_A_NULL;
    assert!(
        (delta - want).abs() < 1e-3,
        "arm A ΔOFV: ferx {delta}, NONMEM {want}"
    );
}

#[test]
fn a_multiplicative_and_an_additive_relation_on_one_parameter_match_nonmem() {
    // Realised worst relative PRED error: 2.65e-15 (bound 30× above it).
    let rel = worst_pred_error(&arm_b(0.02), "additive_cov_mixed.tab");
    assert!(rel < 1e-13, "arm B PRED: worst relative error {rel:e}");

    // Realised difference between the deltas: 3.152e-4 on a delta of +1.0076
    // (3.1e-4 relative); bound 3× the measurement.
    let delta = ferx_ofv(&arm_b(0.02)) - ferx_ofv(&arm_b(0.0));
    let want = NM_OFV_B - NM_OFV_B_NULL;
    assert!(
        (delta - want).abs() < 1e-3,
        "arm B ΔOFV: ferx {delta}, NONMEM {want}"
    );
}

#[test]
fn the_null_twin_is_the_model_with_no_additive_term_at_all() {
    // The differential above is only meaningful if ferx's "slope = 0" arm is
    // genuinely the null model — i.e. if null-at-zero holds all the way through
    // the objective, not just in the generated text. A Pharmpy-verbatim
    // template would add `1` to CL here and this would fail.
    let bare = format!("{HEAD}{TAIL}");
    assert_eq!(
        ferx_ofv(&arm_a(0.0)),
        ferx_ofv(&bare),
        "an additive relation at θ = 0 must be bit-identical to no relation at all"
    );
}

#[test]
fn the_additive_term_is_what_is_being_anchored() {
    // A green anchor is only evidence if the anchored quantity is sensitive to
    // the thing under test. Zeroing the additive slope — and nothing else —
    // must miss the same NONMEM tables by orders of magnitude more than the
    // bound the two anchors above pass at.
    let rel_a = worst_pred_error(&arm_a(0.0), "additive_cov_lin.tab");
    assert!(
        rel_a > 0.1,
        "arm A: with the additive slope at zero the anchor must fail loudly, got {rel_a:e}"
    );
    let rel_b = worst_pred_error(&arm_b(0.0), "additive_cov_mixed.tab");
    assert!(
        rel_b > 0.1,
        "arm B: with the additive slope at zero the anchor must fail loudly, got {rel_b:e}"
    );
}
