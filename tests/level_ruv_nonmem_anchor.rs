//! NONMEM anchor for a θ level block on the residual SD next to `iiv_on_ruv`
//! (#1879).
//!
//! `DV ~ proportional(PROP * exp(ERRSCALE))` with `theta ERRSCALE[STUDY, OCC]`,
//! one study per subject, three occasions each, next to `iiv_on_ruv = ETA_RUV`.
//! `ETA_RUV` scales every record of its subject alike, which is the common
//! shift of the subject's three levels. Before #1879 the span certificate read
//! the predictions alone, where `ETA_RUV` has a zero column, so auto took global
//! `sum_to_zero`: per-study shifts stayed free and the model bound aliased, with
//! `ETA_RUV`'s ω driven to the optimizer's floor. Now the certificate also reads
//! the residual channel, finds the shift, and auto takes `sum_to_zero_within`.
//! This is the fit that #1879 newly accepts, so it is anchored.
//!
//! The kit (`nonmem_anchor/`):
//!
//! - `level_ruv.csv` — four studies of one subject each, three occasions of
//!   three records each, simulated by `simulate_level_ruv.py` (seed 1879);
//! - `level_ruv.ctl` — `$PRED`, the block hand-coded sum-to-zero within each
//!   study (the third occasion is minus the sum of the first two),
//!   `Y = IPRED + IPRED*EXP(L)*EXP(ETA(2))*EPS(1)`, FOCE-I at `MAXEVAL=0` with
//!   `POSTHOC`, `FORMAT=s1PE23.16`. Results in
//!   `results/level_ruv.{lst,ext,phi,tab}` from the `nm3` anchor image, run by
//!   hand since `nm3` cannot baseline a `MAXEVAL=0` run;
//! - `level_ruv_fit.ferx` — the level spelling, which this test binds, so the
//!   binder's verdict is part of the anchor.
//!
//! ferx is evaluated (`outer_maxiter = 0`) at NONMEM's θ, Ω and σ: objective,
//! each subject's η̂ (both random effects), and IPRED at every record.
//!
//! Non-degenerate: every level, the dependent ones included (0.30, −0.10,
//! −0.20; −0.25, 0.40, −0.15; 0.20, 0.10, −0.30; 0.50, −0.05, −0.45), is nonzero
//! and at least 0.05 from the others, so a level read at another level's θ moves
//! the objective; the four `ETA_RUV` η̂ are nonzero and distinct, so the
//! `exp(2·η)` scale is live on every subject; and the residual SD varies across
//! each subject's records, so the level is not a constant `ETA_RUV` would carry.
//! Binder, f64 predictor and the FOCE-I inner loop: no gradient is compared.

use ferx_core::parser::model_parser::LevelContrast;
use ferx_core::types::FitOptions;
use std::collections::HashMap;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn results(name: &str) -> String {
    std::fs::read_to_string(root().join("nonmem_anchor/results").join(name)).unwrap()
}

/// The `-1000000000` row of the `.ext`, by column name.
fn ext() -> HashMap<String, f64> {
    let text = results("level_ruv.ext");
    let mut names: Vec<String> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first() {
            Some(&"ITERATION") => names = f.iter().map(|s| s.to_string()).collect(),
            Some(&"-1000000000") => {
                return names
                    .iter()
                    .cloned()
                    .zip(f.iter().map(|v| v.parse::<f64>().unwrap()))
                    .collect()
            }
            _ => {}
        }
    }
    panic!("no final row in level_ruv.ext");
}

/// Per subject, NONMEM's `(ETA(1), ETA(2))` from the `.phi`.
fn phi() -> Vec<[f64; 2]> {
    results("level_ruv.phi")
        .lines()
        .skip(2)
        .map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .skip(2)
                .take(2)
                .map(|v| v.parse().unwrap())
                .collect();
            [f[0], f[1]]
        })
        .collect()
}

/// IPRED per record, in data order, from the `$TABLE`.
fn ipred() -> Vec<f64> {
    let text = results("level_ruv.tab");
    let mut lines = text.lines().skip(1);
    let header: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    let i = header.iter().position(|h| *h == "IPRED").unwrap();
    lines
        .map(|l| l.split_whitespace().nth(i).unwrap().parse().unwrap())
        .collect()
}

#[test]
fn a_residual_level_block_next_to_iiv_on_ruv_matches_nonmem() {
    let est = ext();
    let prep = ferx_core::prepare_run(
        root()
            .join("nonmem_anchor/level_ruv_fit.ferx")
            .to_str()
            .unwrap(),
        Some(root().join("nonmem_anchor/level_ruv.csv").to_str().unwrap()),
    )
    .unwrap_or_else(|e| panic!("{e}"));

    // The binder's verdict is part of the anchor: within each study, and ferx's
    // θ are NONMEM's in order.
    assert_eq!(
        prep.parsed.bindings.levels["ERRSCALE"].contrast,
        LevelContrast::SumToZeroWithin
    );
    assert_eq!(
        prep.parsed.model.theta_names,
        [
            "TVA",
            "TVK",
            "ERRSCALE[STUDY=1,OCC=1]",
            "ERRSCALE[STUDY=1,OCC=2]",
            "ERRSCALE[STUDY=2,OCC=1]",
            "ERRSCALE[STUDY=2,OCC=2]",
            "ERRSCALE[STUDY=3,OCC=1]",
            "ERRSCALE[STUDY=3,OCC=2]",
            "ERRSCALE[STUDY=4,OCC=1]",
            "ERRSCALE[STUDY=4,OCC=2]",
        ]
    );

    let mut params = prep.init_params.clone();
    params.theta = (1..=10).map(|k| est[&format!("THETA{k}")]).collect();
    // The .ferx Ω and σ² are NONMEM's, to the last bits the .ext prints.
    for (got, want) in [
        (params.omega.matrix[(0, 0)], est["OMEGA(1,1)"]),
        (params.omega.matrix[(1, 1)], est["OMEGA(2,2)"]),
        (params.sigma.values[0].powi(2), est["SIGMA(1,1)"]),
    ] {
        assert!((got - want).abs() < 1e-15, "{got} vs NONMEM {want}");
    }
    let opts = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: false,
        inner_tol: 1e-10,
        ..prep.parsed.fit_options.clone()
    };
    let fit = ferx_core::fit(&prep.parsed.model, &prep.population, &params, &opts)
        .unwrap_or_else(|e| panic!("{e}"));

    let (etas, rows) = (phi(), ipred());
    assert_eq!(etas.len(), 4);
    assert_eq!(fit.subjects.len(), 4);
    let d_ofv = (fit.ofv - est["OBJ"]).abs();
    assert!(
        d_ofv.is_finite(),
        "OFV: ferx {} NONMEM {}",
        fit.ofv,
        est["OBJ"]
    );
    let mut d_eta = 0.0f64;
    for (s, want) in fit.subjects.iter().zip(&etas) {
        for (k, w) in want.iter().enumerate() {
            let d = (s.eta[k] - w).abs();
            assert!(d.is_finite(), "{}: η̂{k} {} vs {w}", s.id, s.eta[k]);
            d_eta = d_eta.max(d);
        }
    }
    let got: Vec<f64> = fit
        .subjects
        .iter()
        .flat_map(|s| s.ipred.iter().copied())
        .collect();
    assert_eq!(got.len(), rows.len(), "records");
    let mut d_ipred = 0.0f64;
    for (g, w) in got.iter().zip(&rows) {
        let d = (g - w).abs();
        assert!(d.is_finite(), "IPRED {g} vs {w}");
        d_ipred = d_ipred.max(d);
    }
    eprintln!(
        "ferx vs NONMEM: |ΔOFV| {d_ofv:.3e} (ferx {:.12}, NONMEM {:.12}), max |Δη̂| \
         {d_eta:.3e}, max |ΔIPRED| {d_ipred:.3e}",
        fit.ofv, est["OBJ"]
    );
    // Realised, Linux aarch64 debug (`tools/linux-test.sh`, the reference platform):
    // |ΔOFV| 6.621e-8, |Δη̂| 2.046e-8, |ΔIPRED| 1.672e-10 — the same in every printed
    // digit as macOS arm64. The model is nonlinear in both η, so the gap is the two
    // inner optimizers' η̂ (IPRED reads ETA_K alone, hence its smaller gap). The
    // bounds leave 15×, 49× and 60× headroom.
    assert!(d_ofv < 1e-6, "OFV: {d_ofv:e}");
    assert!(d_eta < 1e-6, "η̂: {d_eta:e}");
    assert!(d_ipred < 1e-8, "IPRED: {d_ipred:e}");
}
