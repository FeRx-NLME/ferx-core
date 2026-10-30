//! NONMEM anchor for the individual survival curve of a joint PK-TTE model at the EBEs
//! (#1885, test T10) — the first ODE-side TTE anchor with a **live** η.
//!
//! Reference: `nonmem_anchor/pktte_ebe.ctl`, NONMEM 7.6.0 `ADVAN13 TOL=9`, every THETA,
//! OMEGA (0.09, live) and SIGMA `FIX`, `MAXEVAL=0 POSTHOC`, on `nonmem_anchor/pktte_ebe.csv`
//! (8 subjects, oral 100 at 0/12/24 so the later doses land on residual drug, 8 PK
//! observations and one exact or right-censored event each, EVID=2 rows on 2..48).
//! `#OBJV = 120.919689`. Outputs in `nonmem_anchor/results/pktte_ebe.*`. The ferx twin is
//! `nonmem_anchor/pktte_ebe_fit.ferx`.
//!
//! Three comparisons, each isolating one thing:
//!
//!   * (i) ferx's curve at **NONMEM's** ETA1 vs NONMEM's `CHZ`/`HAZ` on every grid and
//!     event row — the predictor alone, with the η taken off the table;
//!   * (ii) ferx's η̂ (a `maxiter = 0` fit, so the inner problem alone) vs NONMEM's ETA1;
//!   * (iii) ferx's curve at its **own** η̂ vs NONMEM's `CHZ` — the whole user path.
//!
//! And the non-degeneracy check: the η = 0 curve, which is what a predictor that ignored
//! the supplied η would return, misses NONMEM's `CHZ` by orders of magnitude more than
//! (i)'s bound at the rows where the η matters.
//!
//! Bounds are measured — see each assertion.

#![cfg(feature = "survival")]

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{FitOptions, ParsedModel, Population};
use ferx_core::{fit, predict_survival_at_etas_diag, SuppliedEtas, SurvivalPredictionResult};

const MODEL: &str = "nonmem_anchor/pktte_ebe_fit.ferx";
const DATA: &str = "nonmem_anchor/pktte_ebe.csv";
const TAB: &str = "nonmem_anchor/results/pktte_ebe.tab";

fn load() -> (ParsedModel, Population) {
    let src = std::fs::read_to_string(MODEL).expect("anchor model file");
    let parsed = parse_full_model(&src).expect("anchor model must parse");
    let (pop, _) = read_population_for(&parsed.model, &None, DATA, None, None, None, &[])
        .expect("endpoint-routed load must succeed");
    (parsed, pop)
}

/// One `$TABLE` row: `(id, time, chz, haz, eta1)`, for every EVID=2 grid row and every
/// TTE record.
fn nonmem_rows() -> Vec<(String, f64, f64, f64, f64)> {
    std::fs::read_to_string(TAB)
        .expect("NONMEM table")
        .lines()
        .skip(2)
        .filter_map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            // ID TIME EVID CMT DV IPRED CHZ HAZ ETA1 MDV
            let keep = f.len() >= 10 && (f[2] == 2.0 || (f[2] == 0.0 && f[3] == 3.0));
            keep.then(|| (format!("{}", f[0] as i64), f[1], f[6], f[7], f[8]))
        })
        .collect()
}

fn nonmem_eta(rows: &[(String, f64, f64, f64, f64)]) -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = Vec::new();
    for (id, _, _, _, eta) in rows {
        if !out.iter().any(|(i, _)| i == id) {
            out.push((id.clone(), *eta));
        }
    }
    out
}

fn grid(rows: &[(String, f64, f64, f64, f64)]) -> Vec<f64> {
    let mut g: Vec<f64> = rows.iter().map(|r| r.1).collect();
    g.sort_by(f64::total_cmp);
    g.dedup();
    g
}

fn curve(
    parsed: &ParsedModel,
    pop: &Population,
    etas: &[(String, f64)],
    grid: &[f64],
) -> Vec<SurvivalPredictionResult> {
    // Supplied in the reverse of population order: a positional mix-up fails here.
    let rev: Vec<&(String, f64)> = etas.iter().rev().collect();
    let supplied = SuppliedEtas::from_table(
        rev.iter().map(|(id, _)| id.clone()).collect(),
        vec!["ETA_CL".into()],
        rev.iter().map(|(_, e)| vec![*e]).collect(),
    )
    .expect("table");
    let m = &parsed.model;
    predict_survival_at_etas_diag(m, pop, &m.default_params, &supplied, grid)
        .expect("individual curves")
        .results
}

/// Worst relative `(H, h)` error of `pred` against NONMEM over `rows`, refusing a
/// non-finite prediction first (a NaN folded through `f64::max` would vanish).
fn worst(pred: &[SurvivalPredictionResult], rows: &[(String, f64, f64, f64, f64)]) -> (f64, f64) {
    let (mut wh, mut wz) = (0.0f64, 0.0f64);
    for (id, t, chz, haz, _) in rows {
        let r = pred
            .iter()
            .find(|r| &r.id == id && r.time == *t)
            .unwrap_or_else(|| panic!("subject {id}: t = {t} is on the grid"));
        assert!(
            r.cum_hazard.is_finite() && r.hazard.is_finite(),
            "subject {id} at t = {t}: non-finite H/h ({}, {})",
            r.cum_hazard,
            r.hazard
        );
        wh = wh.max(((r.cum_hazard - chz) / chz).abs());
        wz = wz.max(((r.hazard - haz) / haz).abs());
    }
    (wh, wz)
}

/// ferx's η̂: a fit with everything FIX and `maxiter = 0`.
fn ferx_eta(parsed: &ParsedModel, pop: &Population) -> (Vec<(String, f64)>, Vec<String>) {
    let opts = FitOptions {
        run_covariance_step: false,
        inner_tol: 1e-10,
        ..parsed.fit_options.clone()
    };
    let m = &parsed.model;
    let r = fit(m, pop, &m.default_params, &opts).expect("inner-only fit");
    (
        r.subjects
            .iter()
            .map(|s| (s.id.clone(), s.eta[0]))
            .collect(),
        r.warnings,
    )
}

#[test]
fn individual_curve_at_nonmems_eta_matches_nonmem() {
    let (parsed, pop) = load();
    let rows = nonmem_rows();
    assert_eq!(
        rows.len(),
        8 * 25,
        "24 grid rows and one event row per subject"
    );
    let eta = nonmem_eta(&rows);
    assert!(
        eta.iter().all(|(_, e)| e.abs() > 1e-3),
        "every ETA1 is live: {eta:?}"
    );
    let g = grid(&rows);

    // (i)
    let (wh, wz) = worst(&curve(&parsed, &pop, &eta, &g), &rows);
    eprintln!("T10(i) worst rel H {wh:.3e}, h {wz:.3e}");
    // Measured 4.42e-9 (H) and 6.07e-9 (h), macOS arm64; bound ~10x above.
    assert!(
        wh < 5e-8,
        "(i) H vs NONMEM CHZ: worst rel {wh:.3e} (measured 4.42e-9)"
    );
    assert!(
        wz < 5e-8,
        "(i) h vs NONMEM HAZ: worst rel {wz:.3e} (measured 6.07e-9)"
    );

    // Non-degeneracy: the η = 0 curve misses by far more than (i)'s bound.
    let zero: Vec<(String, f64)> = eta.iter().map(|(i, _)| (i.clone(), 0.0)).collect();
    let (zh, _) = worst(&curve(&parsed, &pop, &zero, &g), &rows);
    eprintln!("T10 eta=0 curve worst rel H {zh:.3e}");
    assert!(zh > 0.1, "the η = 0 curve must miss NONMEM: {zh:.3e}");
}

#[test]
fn ebes_and_the_curve_at_them_match_nonmem() {
    let (parsed, pop) = load();
    let rows = nonmem_rows();
    let nm = nonmem_eta(&rows);
    let (fx, warnings) = ferx_eta(&parsed, &pop);
    eprintln!("T10 fit warnings: {warnings:#?}");
    assert_eq!(fx.len(), nm.len());

    // (ii)
    let mut worst_eta = 0.0f64;
    for (id, e) in &nm {
        let (_, f) = fx.iter().find(|(i, _)| i == id).expect("fitted subject");
        assert!(f.is_finite(), "subject {id}: η̂ {f}");
        eprintln!("T10(ii) subject {id}: NONMEM {e:+.9} ferx {f:+.9}");
        worst_eta = worst_eta.max((f - e).abs());
    }
    eprintln!("T10(ii) worst |η̂ − ETA1| {worst_eta:.3e}");
    // Measured 4.69e-10 at inner_tol 1e-10: no live-η gap on this model.
    assert!(
        worst_eta < 1e-8,
        "(ii) worst |η̂ − ETA1| {worst_eta:.3e} (measured 4.69e-10)"
    );

    // (iii)
    let (wh, wz) = worst(&curve(&parsed, &pop, &fx, &grid(&rows)), &rows);
    eprintln!("T10(iii) worst rel H {wh:.3e}, h {wz:.3e}");
    // Measured 4.47e-9 (H) and 7.11e-9 (h); bound ~10x above.
    assert!(
        wh < 5e-8,
        "(iii) H at ferx η̂ vs NONMEM CHZ: worst rel {wh:.3e} (measured 4.47e-9)"
    );
    assert!(
        wz < 5e-8,
        "(iii) h at ferx η̂ vs NONMEM HAZ: worst rel {wz:.3e} (measured 7.11e-9)"
    );
}
