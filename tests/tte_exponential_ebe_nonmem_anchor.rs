//! NONMEM anchor for the individual survival curve of an **analytic** hazard at the EBEs
//! (#1885, test T11).
//!
//! Reference: `nonmem_anchor/tte_exponential_ebe.ctl`, NONMEM 7.6.0, `MAXEVAL=0 POSTHOC`
//! at the Laplace optimum of `tte_exponential_cov_focei.ctl` (THETA1 = 3.32303E-02,
//! OMEGA(1,1) = 2.84734E-01, both FIX) on `data/tte_exponential.csv` (30 subjects, one exact
//! or right-censored record each). `#OBJV = 166.439575886`, the `_focei` run's final OFV to
//! the 9th digit, so the evaluation sits at that optimum. Output in
//! `nonmem_anchor/results/tte_exponential_ebe.tab`.
//!
//! The same three comparisons as `tests/pktte_ebe_nonmem_anchor.rs`: (i) ferx's `H` at
//! NONMEM's ETA1, (ii) ferx's η̂ vs ETA1, (iii) ferx's `H` at its own η̂, each against
//! NONMEM's `CHZ` at the subject's record; plus the η = 0 curve, which must miss.

#![cfg(feature = "survival")]

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{CompiledModel, FitOptions, ModelParameters, OmegaMatrix, Population};
use ferx_core::{fit, predict_survival_at_etas_diag, SuppliedEtas};

const TAB: &str = "nonmem_anchor/results/tte_exponential_ebe.tab";

fn load() -> (CompiledModel, Population, ModelParameters) {
    let src = std::fs::read_to_string("examples/tte_exponential.ferx").expect("model file");
    let m = parse_model_string(&src).expect("parse");
    let (pop, _) =
        read_population_for(&m, &None, "data/tte_exponential.csv", None, None, None, &[])
            .expect("data");
    let mut p = m.default_params.clone();
    p.theta[0] = 3.32303e-2;
    p.omega = OmegaMatrix::from_diagonal(&[2.84734e-1], vec!["ETA_LAMBDA".into()]);
    (m, pop, p)
}

/// `(id, time, chz, eta1)` per subject.
fn nonmem_rows() -> Vec<(String, f64, f64, f64)> {
    std::fs::read_to_string(TAB)
        .expect("NONMEM table")
        .lines()
        .skip(2)
        .filter_map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            // ID TIME DV CHZ ETA1
            (f.len() >= 5).then(|| (format!("{}", f[0] as i64), f[1], f[3], f[4]))
        })
        .collect()
}

/// Worst relative `H` error at each subject's record, with each subject at `etas[id]`.
/// The population is reversed, so a positional lookup gives another subject's η.
fn worst_h(
    m: &CompiledModel,
    pop: &Population,
    p: &ModelParameters,
    etas: &[(String, f64)],
    rows: &[(String, f64, f64, f64)],
) -> f64 {
    let supplied = SuppliedEtas::from_table(
        etas.iter().map(|(id, _)| id.clone()).collect(),
        vec!["ETA_LAMBDA".into()],
        etas.iter().map(|(_, e)| vec![*e]).collect(),
    )
    .expect("table");
    let mut rev = pop.clone();
    rev.subjects.reverse();
    let mut grid: Vec<f64> = rows.iter().map(|r| r.1).collect();
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    let out = predict_survival_at_etas_diag(m, &rev, p, &supplied, &grid).expect("curves");
    let mut w = 0.0f64;
    for (id, t, chz, _) in rows {
        let r = out
            .results
            .iter()
            .find(|r| &r.id == id && r.time == *t)
            .expect("record time on the grid");
        assert!(r.cum_hazard.is_finite(), "subject {id}: H {}", r.cum_hazard);
        w = w.max(((r.cum_hazard - chz) / chz).abs());
    }
    w
}

#[test]
fn individual_cum_hazard_at_the_ebes_matches_nonmem() {
    let (m, pop, p) = load();
    let rows = nonmem_rows();
    assert_eq!(rows.len(), 30);
    assert!(rows.iter().all(|r| r.3.abs() > 1e-3), "every ETA1 is live");
    let nm: Vec<(String, f64)> = rows.iter().map(|r| (r.0.clone(), r.3)).collect();

    // (i)
    let wi = worst_h(&m, &pop, &p, &nm, &rows);
    eprintln!("T11(i) worst rel H {wi:.3e}");
    // Measured 2.09e-9 on Linux aarch64 and macOS arm64, the same digits: the
    // 9-significant-figure print floor of the table.
    assert!(
        wi < 2e-8,
        "(i) H at NONMEM's ETA1 vs CHZ: worst rel {wi:.3e} (measured 2.09e-9)"
    );

    // η = 0 must miss.
    let zero: Vec<(String, f64)> = nm.iter().map(|(i, _)| (i.clone(), 0.0)).collect();
    let wz = worst_h(&m, &pop, &p, &zero, &rows);
    eprintln!("T11 eta=0 worst rel H {wz:.3e}");
    assert!(wz > 0.1, "the η = 0 curve must miss NONMEM: {wz:.3e}");

    // (ii): ferx's η̂ at the same point, inner problem only.
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        inner_tol: 1e-10,
        verbose: false,
        ..FitOptions::default()
    };
    let r = fit(&m, &pop, &p, &opts).expect("inner-only fit");
    eprintln!("T11 fit warnings: {:#?}", r.warnings);
    let fx: Vec<(String, f64)> = r
        .subjects
        .iter()
        .map(|s| (s.id.clone(), s.eta[0]))
        .collect();
    let mut we = 0.0f64;
    for (id, e) in &nm {
        let (_, f) = fx.iter().find(|(i, _)| i == id).expect("fitted subject");
        assert!(f.is_finite(), "subject {id}: η̂ {f}");
        we = we.max((f - e).abs());
    }
    eprintln!("T11(ii) worst |η̂ − ETA1| {we:.3e}");
    // Measured 5.97e-9 at inner_tol 1e-10, Linux aarch64 and macOS arm64 alike; bound ~10x
    // above.
    assert!(
        we < 6e-8,
        "(ii) worst |η̂ − ETA1| {we:.3e} (measured 5.97e-9)"
    );

    // (iii)
    let wiii = worst_h(&m, &pop, &p, &fx, &rows);
    eprintln!("T11(iii) worst rel H {wiii:.3e}");
    // Measured 5.90e-9, Linux aarch64 and macOS arm64 alike; bound ~10x above.
    assert!(
        wiii < 6e-8,
        "(iii) H at ferx η̂ vs CHZ: worst rel {wiii:.3e} (measured 5.90e-9)"
    );
}
