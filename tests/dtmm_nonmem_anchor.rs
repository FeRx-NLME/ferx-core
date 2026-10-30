//! NONMEM 7.5.1 anchor for the discrete-time Markov (DTMM) endpoint, `[markov_model]
//! type = dtmm` (#2473).
//!
//! Two mixed-effects DTMMs on seeded simulated data (`nonmem_anchor/make_dtmm_fixtures.py`),
//! each fitted by NONMEM with a `PREVIOUS`-DV `$PRED` under `$EST METHOD=COND LAPLACE LIKE`
//! (`nonmem_anchor/dtmm_{2,3}state.ctl`, outputs in `nonmem_anchor/results/`):
//!
//! - `dtmm_2state` — awake/asleep, a time-constant covariate on one transition and an η on
//!   each;
//! - `dtmm_3state` — three ordered states as a cumulative logit conditioned on the previous
//!   state, so four of the six steps are a *difference* of logits and every stay probability
//!   is the diagonal ferx fills in.
//!
//! Both datasets space their records irregularly and carry `MDV = 1` rows inside the chains,
//! so the two conventions ferx documents — a step is a record whatever its `Δt`, and a
//! missing state links to the last observed one — are live on both sides: NONMEM reads the
//! previous observed DV from a `PDV` column, ferx derives it from the record order.
//!
//! The Tier-2 tests evaluate ferx's objective once (`maxiter = 0`) at NONMEM's final
//! estimates (`nonmem_anchor/dtmm_{2,3}state_fit.ferx`) and compare it to NONMEM's. The
//! slow test fits ferx to convergence from NONMEM's initial estimates.
//!
//! | fixture | NONMEM OFV | exact Laplace | ferx at NONMEM's θ | ferx − NONMEM | ferx − exact |
//! |---------|-----------:|--------------:|-------------------:|--------------:|-------------:|
//! | `dtmm_2state` | 1049.4031789 | 1049.4079889 | 1049.4077891 | 4.61e-3 | 2.00e-4 |
//! | `dtmm_3state` | 1126.0809184 | — | 1126.0808614 | 5.70e-5 | — |
//!
//! Linux x86_64, glibc 2.41 (`tools/linux-test.sh`, `--profile ci-test`) — the reference
//! platform the bounds below are measured on. macOS arm64 differs in the 7th significant
//! figure (1049.4077942 / 1126.0808629: |ΔOFV| 4.62e-3 / 5.55e-5, 1.95e-4 to exact), inside
//! every bound with the same headroom. The 2-state gap to NONMEM was chased, not widened over: the model is
//! separable (each η enters one transition), so its Laplace objective has an exact
//! closed-form evaluation, `nonmem_anchor/dtmm_2state_exact_laplace.py` — exact EBEs and an
//! exact Hessian, outside both engines. ferx sits 2.0e-4 from it; NONMEM sits 4.8e-3 from it,
//! almost all on two subjects (48, 11) with at most two awake steps under `ω²_AS = 0.063`,
//! where an η is prior-dominated and NONMEM LAPLACE's numerical second derivatives are least
//! accurate (NONMEM's η̂ match the exact ones to ~5e-9 on the subjects checked, so the gap is
//! in the Hessian term, not the mode). So the 2-state NONMEM bound
//! is set from that measured gap, and ferx is held tightly to the exact value instead.
//!
//! Non-degeneracy: the objective moves by ~300 when the previous state is ignored — see
//! `previous_state_is_load_bearing`.

#![cfg(feature = "markov")]

use std::path::PathBuf;

use ferx_core::{fit, prepare_run};

fn anchor(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("nonmem_anchor");
    p.push(name);
    p.to_string_lossy().into_owned()
}

/// The objective at NONMEM's final row (`-1000000000`) of `results/<stem>.ext`.
fn nm_ofv(stem: &str) -> f64 {
    let text = std::fs::read_to_string(anchor(&format!("results/{stem}.ext")))
        .unwrap_or_else(|e| panic!("results/{stem}.ext: {e}"));
    let mut lines = text.lines().filter(|l| !l.starts_with("TABLE NO."));
    let header: Vec<&str> = lines.next().expect("header").split_whitespace().collect();
    let obj = header.iter().position(|h| *h == "OBJ").expect("OBJ column");
    lines
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .find(|r| r.first() == Some(&"-1000000000"))
        .map(|r| r[obj].parse().expect("numeric OBJ"))
        .unwrap_or_else(|| panic!("{stem}.ext: no final row"))
}

/// ferx's objective for `nonmem_anchor/<model>.ferx` on `nonmem_anchor/<data>.csv`.
fn ferx_ofv(model: &str, data: &str) -> f64 {
    let run = prepare_run(
        &anchor(&format!("{model}.ferx")),
        Some(&anchor(&format!("{data}.csv"))),
    )
    .unwrap_or_else(|e| panic!("{model}: the anchor model and dataset load: {e}"));
    let r = fit(
        &run.parsed.model,
        &run.population,
        &run.init_params,
        &run.parsed.fit_options,
    )
    .unwrap_or_else(|e| panic!("{model}: the evaluation runs: {e}"));
    assert!(r.ofv.is_finite(), "{model}: OFV {}", r.ofv);
    r.ofv
}

/// 2-state vs NONMEM: measured 4.61e-3 (Linux), the NONMEM numerical-Hessian error explained in the
/// module docs; 2× headroom, still ~3e4 below the previous-state effect.
const OFV_TOL_NM_2STATE: f64 = 1e-2;
/// 2-state vs the exact Laplace value: measured 2.00e-4 (Linux; ferx's FD Laplace Hessian); 5×.
const OFV_TOL_EXACT_2STATE: f64 = 1e-3;
/// `dtmm_2state_exact_laplace.py` at NONMEM's final estimates.
const EXACT_LAPLACE_2STATE: f64 = 1049.4079889254103;
/// 3-state vs NONMEM: measured 5.70e-5 (Linux); 8.8×.
const OFV_TOL_NM_3STATE: f64 = 5e-4;

fn report(label: &str, ferx: f64, nm: f64) -> f64 {
    let d = (ferx - nm).abs();
    eprintln!("{label}: ferx {ferx:.10} NONMEM {nm:.10} |dOFV| {d:.3e}");
    d
}

#[test]
fn dtmm_2state_matches_nonmem() {
    let ferx = ferx_ofv("dtmm_2state_fit", "dtmm_2state");
    let d = report("dtmm_2state", ferx, nm_ofv("dtmm_2state"));
    assert!(d <= OFV_TOL_NM_2STATE, "dtmm_2state |dOFV| {d:.3e}");
    let e = (ferx - EXACT_LAPLACE_2STATE).abs();
    eprintln!("dtmm_2state: exact Laplace {EXACT_LAPLACE_2STATE:.10} |dOFV| {e:.3e}");
    assert!(
        e <= OFV_TOL_EXACT_2STATE,
        "dtmm_2state vs exact |dOFV| {e:.3e}"
    );
}

#[test]
fn dtmm_3state_matches_nonmem() {
    let d = report(
        "dtmm_3state",
        ferx_ofv("dtmm_3state_fit", "dtmm_3state"),
        nm_ofv("dtmm_3state"),
    );
    assert!(d <= OFV_TOL_NM_3STATE, "dtmm_3state |dOFV| {d:.3e}");
}

/// The anchor is not satisfied by an objective that ignores the previous state: shuffling
/// which state a step starts from (here, scoring the 3-state fixture with every row's
/// previous-state shift removed) moves the OFV by far more than any bound here. Done by editing
/// the model, not the data, so the comparison stays on the anchored dataset.
#[test]
fn previous_state_is_load_bearing() {
    let src = std::fs::read_to_string(anchor("dtmm_3state_fit.ferx")).unwrap();
    let flat = src
        .replace(" - G1 + ETA", " + ETA")
        .replace(" - G2 + ETA", " + ETA");
    assert_ne!(src, flat);
    let dir = std::env::temp_dir().join(format!("ferx_dtmm_flat_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("flat.ferx");
    std::fs::write(&model, flat).unwrap();
    let run = prepare_run(model.to_str().unwrap(), Some(&anchor("dtmm_3state.csv"))).expect("load");
    let r = fit(
        &run.parsed.model,
        &run.population,
        &run.init_params,
        &run.parsed.fit_options,
    )
    .expect("evaluation runs");
    let _ = std::fs::remove_dir_all(&dir);
    let gap = (r.ofv - nm_ofv("dtmm_3state")).abs();
    eprintln!("previous state dropped: OFV {:.6}, gap {gap:.3e}", r.ofv);
    assert!(gap > 1e4 * OFV_TOL_NM_3STATE, "gap {gap:.3e}");
}

/// Fitted to convergence from NONMEM's initial estimates, ferx reaches at least NONMEM's
/// optimum, with the same fixed effects.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn dtmm_2state_fit_converges_to_nonmem() {
    let run = prepare_run(
        &format!("{}/examples/dtmm_2state.ferx", env!("CARGO_MANIFEST_DIR")),
        Some(&anchor("dtmm_2state.csv")),
    )
    .expect("load");
    let r = fit(
        &run.parsed.model,
        &run.population,
        &run.init_params,
        &run.parsed.fit_options,
    )
    .expect("fit");
    let nm = nm_ofv("dtmm_2state");
    eprintln!(
        "dtmm_2state fit: OFV {:.6} θ {:?} ω {:?}",
        r.ofv, r.theta, r.omega
    );
    // The objective is flat along the ω split: NONMEM moved ω²_AS 0.22 → 0.06 for 0.9 OFV
    // between two stopping rules, and ferx goes further, to 1046.894167 at ω² = (0.185,
    // 0.092) on Linux x86_64 (macOS arm64: 1046.893878), 2.5 below NONMEM's stopping point.
    // That lower value is real, not an FD-Laplace artifact: `dtmm_2state_exact_laplace.py`
    // at ferx's Linux estimates gives 1046.894052. So the comparison is one-sided on the
    // objective, and on the fixed effects (measured |Δθ| 0.001 / 0.026 / 0.011 on Linux).
    assert!(r.ofv <= nm + 0.05, "ferx OFV {} vs NONMEM {nm}", r.ofv);
    let nm_theta = [
        -1.0065933892463419,
        0.88239637849927277,
        -0.64393697409745099,
    ];
    for (i, (f, n)) in r.theta.iter().zip(nm_theta).enumerate() {
        assert!((f - n).abs() < 0.05, "θ{i}: ferx {f} NONMEM {n}");
    }
}
