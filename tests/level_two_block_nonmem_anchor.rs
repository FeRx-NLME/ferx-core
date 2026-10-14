//! NONMEM anchor for a model with two θ level blocks (#1963).
//!
//! `E0 = TVE0 * (1 + 0.01 * TIME^2) + B2 + PLACEBO + ETA_E0`, with `theta B2[OCC]`
//! (three arms, absorbing nothing) and `theta PLACEBO[STUDY]` (one study per subject,
//! absorbing `ETA_E0`), both resolving to sum-to-zero. Before #1963 the binder counted
//! B2 with all its levels, which sum to a constant and so carried PLACEBO's mean, and
//! refused the model "under any contrast". It is the same model as B2 and PLACEBO
//! written out as θ of their coding, which bound — and which is what NONMEM can spell.
//!
//! The kit (`nonmem_anchor/`):
//!
//! - `level_two_block.csv` — six studies of one subject each, three arms (`OCC`) at
//!   each of six times, simulated by `simulate_level_two_block.py` (seed 1963);
//! - `level_two_block.ctl` — `$PRED`, both blocks hand-coded sum-to-zero (the last
//!   level minus the others' sum), FOCE-I at `MAXEVAL=0` with `POSTHOC`, every level
//!   nonzero, `FORMAT=s1PE23.16`. Results in `results/level_two_block.{lst,ext,phi,tab}`
//!   from the `nm3` anchor image (`nm3` cannot baseline a `MAXEVAL=0` run, so only that
//!   config ran; see the row in `nonmem_anchor/README.md`);
//! - `level_two_block_fit.ferx` — the level spelling, which this test binds, so the
//!   binder's verdict on the model is part of the anchor;
//! - `level_two_block_hand.ferx` — ferx's own hand spelling, the NONMEM control's
//!   arithmetic in the DSL.
//!
//! ferx is evaluated (`outer_maxiter = 0`) at NONMEM's θ. Two assertions: the level
//! spelling agrees with NONMEM — objective, each subject's η̂, and PRED / IPRED at every
//! record — to the realised error below; and the level spelling equals the hand
//! spelling **bit for bit**, since ferx derives a dependent level as minus the left
//! fold of the free ones, which is what the hand spelling writes.
//!
//! Non-degenerate: within each block every level, the dependent one included
//! (B2: 0.25, −0.15, −0.10; PLACEBO: 0.35, −0.25, 0.15, 0.05, −0.40, 0.10), is nonzero
//! and at least 0.05 from the others, so a level read at another level's θ moves
//! PRED; the six η̂ differ (−0.090 … 0.421), and every arm of every study is observed
//! at every time. Binder plus the f64
//! predictor and the FOCE-I inner loop: no gradient is compared, so no FD/dual engine
//! is under test here.

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
    let text = results("level_two_block.ext");
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
    panic!("no final row in level_two_block.ext");
}

/// Per subject, NONMEM's `ETA(1)` from the `.phi`.
fn phi() -> Vec<f64> {
    results("level_two_block.phi")
        .lines()
        .skip(2)
        .map(|l| l.split_whitespace().nth(2).unwrap().parse().unwrap())
        .collect()
}

/// `(PRED, IPRED)` per record, in data order, from the `$TABLE`.
fn tab() -> Vec<(f64, f64)> {
    let text = results("level_two_block.tab");
    let mut lines = text.lines().skip(1);
    let header: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    let at = |n: &str| header.iter().position(|h| *h == n).unwrap();
    let (p, i) = (at("PRED"), at("IPRED"));
    lines
        .map(|l| {
            let f: Vec<f64> = l.split_whitespace().map(|v| v.parse().unwrap()).collect();
            (f[p], f[i])
        })
        .collect()
}

/// Bind `model` on the anchor data and evaluate it at `theta`.
fn evaluate(model: &str, theta: &[f64]) -> (ferx_core::PreparedRun, ferx_core::types::FitResult) {
    let prep = ferx_core::prepare_run(
        root().join("nonmem_anchor").join(model).to_str().unwrap(),
        Some(
            root()
                .join("nonmem_anchor/level_two_block.csv")
                .to_str()
                .unwrap(),
        ),
    )
    .unwrap_or_else(|e| panic!("{model}: {e}"));
    let mut params = prep.init_params.clone();
    assert_eq!(params.theta.len(), theta.len(), "{model}: θ count");
    params.theta = theta.to_vec();
    let opts = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..prep.parsed.fit_options.clone()
    };
    let f = ferx_core::fit(&prep.parsed.model, &prep.population, &params, &opts)
        .unwrap_or_else(|e| panic!("{model}: {e}"));
    (prep, f)
}

#[test]
fn two_level_blocks_match_nonmem_and_the_hand_spelling() {
    let est = ext();
    let theta: Vec<f64> = (1..=10).map(|k| est[&format!("THETA{k}")]).collect();
    assert_eq!(est["OMEGA(1,1)"], 0.1, "the .ferx ω² is NONMEM's");
    assert_eq!(est["SIGMA(1,1)"], 0.25, "the .ferx σ² is NONMEM's");

    let (prep, level) = evaluate("level_two_block_fit.ferx", &theta);
    // The binder's verdict is part of the anchor: both blocks bind sum-to-zero, and
    // ferx's θ are NONMEM's in order — the three structural, B2's free arms, then
    // PLACEBO's free studies.
    let bound = &prep.parsed.bindings.levels;
    assert_eq!(bound["B2"].contrast, LevelContrast::SumToZero);
    assert_eq!(bound["PLACEBO"].contrast, LevelContrast::SumToZero);
    assert_eq!(
        prep.parsed.model.theta_names,
        [
            "TVE0",
            "TVEMAX",
            "TVET50",
            "B2[OCC=1]",
            "B2[OCC=2]",
            "PLACEBO[STUDY=1]",
            "PLACEBO[STUDY=2]",
            "PLACEBO[STUDY=3]",
            "PLACEBO[STUDY=4]",
            "PLACEBO[STUDY=5]",
        ]
    );

    let (_, hand) = evaluate("level_two_block_hand.ferx", &theta);

    // ferx level spelling against NONMEM.
    let (etas, rows) = (phi(), tab());
    assert_eq!(etas.len(), 6);
    assert_eq!(level.subjects.len(), 6);
    let d_ofv = (level.ofv - est["OBJ"]).abs();
    assert!(
        d_ofv.is_finite(),
        "OFV: ferx {} NONMEM {}",
        level.ofv,
        est["OBJ"]
    );
    let mut d_eta = 0.0f64;
    for (s, want) in level.subjects.iter().zip(&etas) {
        let d = (s.eta[0] - want).abs();
        assert!(d.is_finite(), "{}: η̂ {} vs {want}", s.id, s.eta[0]);
        d_eta = d_eta.max(d);
    }
    let preds: Vec<(f64, f64)> = level
        .subjects
        .iter()
        .flat_map(|s| s.pred.iter().copied().zip(s.ipred.iter().copied()))
        .collect();
    assert_eq!(preds.len(), rows.len(), "records");
    let (mut d_pred, mut d_ipred) = (0.0f64, 0.0f64);
    for ((p, i), (wp, wi)) in preds.iter().zip(&rows) {
        let (dp, di) = ((p - wp).abs(), (i - wi).abs());
        assert!(
            dp.is_finite() && di.is_finite(),
            "PRED {p}/{wp}, IPRED {i}/{wi}"
        );
        d_pred = d_pred.max(dp);
        d_ipred = d_ipred.max(di);
    }
    eprintln!(
        "ferx vs NONMEM: |ΔOFV| {d_ofv:.3e} (ferx {:.12}, NONMEM {:.12}), max |Δη̂| \
         {d_eta:.3e}, max |ΔPRED| {d_pred:.3e}, max |ΔIPRED| {d_ipred:.3e}",
        level.ofv, est["OBJ"]
    );
    // Realised, macOS arm64 debug: |ΔOFV| 5.3e-15, |Δη̂| 5.6e-17, |ΔPRED| 0, |ΔIPRED|
    // 1.8e-15. The model is linear in η with an additive error, so η̂ is a closed form
    // on both sides and needs no inner-optimizer allowance. The bounds leave five
    // decades or more of headroom; a level read at another level's θ moves PRED by at
    // least 0.05.
    assert!(d_ofv < 1e-9, "OFV: {d_ofv:e}");
    assert!(d_eta < 1e-10, "η̂: {d_eta:e}");
    assert!(d_pred < 1e-12, "PRED: {d_pred:e}");
    assert!(d_ipred < 1e-10, "IPRED: {d_ipred:e}");

    // ferx level spelling against ferx hand spelling, bit for bit.
    assert_eq!(
        level.ofv.to_bits(),
        hand.ofv.to_bits(),
        "OFV: level {} hand {}",
        level.ofv,
        hand.ofv
    );
    for (a, b) in level.subjects.iter().zip(&hand.subjects) {
        assert_eq!(a.eta[0].to_bits(), b.eta[0].to_bits(), "{}: η̂", a.id);
        assert!(
            a.pred
                .iter()
                .zip(&b.pred)
                .all(|(x, y)| x.to_bits() == y.to_bits())
                && a.ipred
                    .iter()
                    .zip(&b.ipred)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
            "{}: PRED / IPRED differ",
            a.id
        );
    }
}
