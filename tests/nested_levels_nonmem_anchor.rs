//! NONMEM anchor for nested variability levels (#1082 PR2): oracle O1, and the
//! nesting-by-key oracle O3 on the same data.
//!
//! ## The kit (`nonmem_anchor/nested_levels_3.*`)
//!
//! - `nested_levels_3.csv` — the #1082 design note's synthetic MBMA data (seed 1082):
//!   12 sponsors × 2 studies × 2 arms × 5 visits = 240 arm-mean rows, a nonlinear
//!   Emax time course, `EMAX = TVEMAX·exp(η_sponsor + η_study + κ_arm/√NARM)`, NARM in
//!   {25, …, 400}. `ID` is the sponsor. `ARM` is numbered 1, 2 inside each study;
//!   `ARMID` is the same arm numbered uniquely across the whole file.
//! - `nested_levels_3.ctl` (outputs in `results/`) — the note's `exact3` coding: the study and arm effects are
//!   occasion-style ETA slots with `$OMEGA BLOCK(1) SAME`, the arm scaling written by
//!   hand. FOCEI, converged in NONMEM 7.6.0 (`nm3`, anchor config) to OBJ
//!   107.04357104523862, the same bits as the design note's run.
//! - `nested_levels_3_eval.ctl` — the same model at that run's estimates (as `.ext`
//!   prints them), `MAXEVAL=0 POSTHOC`: OBJ 107.04357106164440 and the EBEs in
//!   `results/nested_levels_3_eval.phi`.
//! - `nested_levels_3.ferx` — the ferx twin: two `level` lines and a weighted kappa at
//!   `ARM`, initial estimates set to the same values.
//!
//! ferx evaluates that point with `outer_maxiter = 0`, so the comparison is two engines
//! scoring one parameter vector, not two optimizers racing.
//!
//! ## Measured (macOS arm64)
//!
//! | quantity | worst \|ferx − NONMEM\| | bound |
//! |---|---|---|
//! | OFV | 9.869e-8 | `1e-6` (10× above) |
//! | η̂ (sponsor), κ̂ (study, arm), 108 values | 8.479e-6 | `2e-5` (2.4× above) |
//!
//! The engine under test is the FD route: a nested model reports finite-difference
//! inner and outer gradients (asserted below), so the EBE search and the κ columns of
//! the FOCEI `H` are central differences of the per-cell prediction.

use ferx_core::api::read_population_for;
use ferx_core::build_info::GradientMethodKind;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, EstimationMethod, FitOptions, FitResult};
use std::path::Path;

/// NONMEM 7.6.0 `OBJ` at the evaluated point (`results/nested_levels_3_eval.ext`, `-1000000000`).
const NM_OFV: f64 = 107.04357106164440;
/// Worst realised |ΔOFV| 9.869e-8 (macOS arm64); `1e-6` is 10× above it. A one-level
/// misreading of the data (κ_study drawn per arm) moves the OFV by whole units.
const OFV_TOL: f64 = 1e-6;
/// Worst realised |Δ| over every η̂ and κ̂ against `.phi`'s six-significant-figure
/// print: 8.479e-6 (macOS arm64), of which up to 5e-6 is the print's own rounding on
/// values up to 3.2; `2e-5` is 2.4× above it.
const EBE_TOL: f64 = 2e-5;

const MODEL: &str = include_str!("../nonmem_anchor/nested_levels_3.ferx");

fn eval(model_src: &str) -> FitResult {
    let parsed = parse_full_model(model_src).expect("anchor model must parse");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "nonmem_anchor/nested_levels_3.csv",
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("anchor data must load");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    fit(&parsed.model, &pop, &parsed.model.default_params, &opts).expect("anchor evaluation")
}

/// One `.phi` row: the subject id and `ETA(1)..ETA(7)`.
fn nonmem_phi() -> Vec<(u32, [f64; 7])> {
    let text = std::fs::read_to_string("nonmem_anchor/results/nested_levels_3_eval.phi").unwrap();
    text.lines()
        .skip(2)
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let mut eta = [0.0; 7];
            for (k, e) in eta.iter_mut().enumerate() {
                *e = f[2 + k].parse().unwrap();
            }
            (f[1].parse().unwrap(), eta)
        })
        .collect()
}

/// Per subject, in file order, the distinct `(STUDY, SLOT)` pairs of its rows in
/// first-row order — the cells ferx reports κ̂ for. NONMEM's study slot is `ETA(1 +
/// STUDY)` and its arm slot `ETA(3 + SLOT)`.
fn cells_per_subject() -> Vec<(u32, Vec<(usize, usize)>)> {
    let text = std::fs::read_to_string("nonmem_anchor/nested_levels_3.csv").unwrap();
    let mut out: Vec<(u32, Vec<(usize, usize)>)> = Vec::new();
    for l in text.lines().skip(1) {
        let f: Vec<&str> = l.split(',').collect();
        let (id, study, slot) = (
            f[0].parse().unwrap(),
            f[1].parse().unwrap(),
            f[5].parse().unwrap(),
        );
        if out.last().is_none_or(|(i, _)| *i != id) {
            out.push((id, Vec::new()));
        }
        let cells = &mut out.last_mut().unwrap().1;
        if !cells.contains(&(study, slot)) {
            cells.push((study, slot));
        }
    }
    out
}

/// O1: ferx's OFV and every EBE at NONMEM's `exact3` optimum.
#[test]
fn nested_levels_match_nonmem_exact3_at_its_optimum() {
    let r = eval(MODEL);
    let d_ofv = (r.ofv - NM_OFV).abs();
    eprintln!(
        "ferx OFV {:.14}  NONMEM {NM_OFV:.14}  |Δ| {d_ofv:.3e}",
        r.ofv
    );
    assert!(
        r.ofv.is_finite() && d_ofv < OFV_TOL,
        "OFV {} vs NONMEM {NM_OFV}",
        r.ofv
    );

    let phi = nonmem_phi();
    let cells = cells_per_subject();
    assert_eq!(phi.len(), 12);
    assert_eq!(r.subjects.len(), 12);
    let mut worst = 0.0f64;
    let mut compared = 0usize;
    for (i, ((id, nm), (cid, cell))) in phi.iter().zip(&cells).enumerate() {
        assert_eq!(id, cid, "subject order");
        // Two studies with two arms each: four cells, four κ̂ vectors `[κ_study, κ_arm]`.
        assert_eq!(cell.len(), 4, "subject {id}: cells {cell:?}");
        let kappas = &r.ebe_kappas[i];
        assert_eq!(kappas.len(), 4, "subject {id}: one κ̂ vector per cell");
        let mut check = |got: f64, want: f64, what: String| {
            assert!(got.is_finite(), "{what}: ferx {got}");
            let d = (got - want).abs();
            assert!(d < EBE_TOL, "{what}: ferx {got} vs NONMEM {want}");
            worst = worst.max(d);
            compared += 1;
        };
        check(
            r.subjects[i].eta[0],
            nm[0],
            format!("subject {id} η_sponsor"),
        );
        for (g, &(study, slot)) in cell.iter().enumerate() {
            check(
                kappas[g][0],
                nm[study],
                format!("subject {id} κ_study {study}"),
            );
            check(
                kappas[g][1],
                nm[2 + slot],
                format!("subject {id} κ_arm slot {slot}"),
            );
        }
    }
    assert_eq!(compared, 12 * 9);
    eprintln!("worst |Δ EBE| {worst:.3e}");

    // The route: every gradient of a nested model is finite-differenced until the
    // analytic walks learn the compact layout.
    assert_eq!(
        r.gradient_method_inner,
        GradientMethodKind::FiniteDifferences.as_str()
    );
    assert_ne!(
        r.gradient_method_outer,
        GradientMethodKind::Analytic.as_str()
    );
}

/// O3, nesting by key: `ARM` repeats 1, 2 in every study, and `ARMID` numbers the same
/// arms uniquely across the file. A group is identified by its whole label path, so
/// the two spellings are the same model and score the same bits. A grouping that read
/// the label alone would merge arm 1 of study 1 with arm 1 of study 2 under `ARM`
/// (one κ_arm for two arms) but not under `ARMID`, and the two OFVs would part.
#[test]
fn arm_labels_local_to_a_study_score_like_globally_unique_ones() {
    // The straddle: under ARM a label recurs across studies, under ARMID it does not.
    let text = std::fs::read_to_string("nonmem_anchor/nested_levels_3.csv").unwrap();
    let mut by_label: std::collections::HashMap<(String, &str), std::collections::HashSet<&str>> =
        Default::default();
    let mut unique: std::collections::HashMap<(String, &str), std::collections::HashSet<&str>> =
        Default::default();
    for l in text.lines().skip(1) {
        let f: Vec<&str> = l.split(',').collect();
        by_label
            .entry((f[0].to_string(), f[2]))
            .or_default()
            .insert(f[1]);
        unique
            .entry((f[0].to_string(), f[3]))
            .or_default()
            .insert(f[1]);
    }
    assert!(by_label.values().any(|studies| studies.len() > 1));
    assert!(unique.values().all(|studies| studies.len() == 1));

    let local = eval(MODEL);
    let global = eval(&MODEL.replace("level ARM by ARM in STUDY", "level ARM by ARMID in STUDY"));
    assert_ne!(
        MODEL,
        MODEL.replace("level ARM by ARM in STUDY", "level ARM by ARMID in STUDY")
    );
    assert!(local.ofv.is_finite());
    assert_eq!(
        local.ofv.to_bits(),
        global.ofv.to_bits(),
        "ARM {} vs ARMID {}",
        local.ofv,
        global.ofv
    );
}

#[test]
fn the_anchor_files_are_where_the_tests_read_them() {
    for f in [
        "nested_levels_3.ctl",
        "results/nested_levels_3.ext",
        "results/nested_levels_3.lst",
        "nested_levels_3_eval.ctl",
        "results/nested_levels_3_eval.ext",
        "results/nested_levels_3_eval.phi",
    ] {
        assert!(Path::new("nonmem_anchor").join(f).exists(), "{f}");
    }
}
