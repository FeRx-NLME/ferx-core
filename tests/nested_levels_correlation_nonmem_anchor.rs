//! NONMEM anchor for nested levels composed with a residual correlation (#1082 PR5, V7):
//! `exact3` with its arms' residuals correlated, `correlation ar1|car1(RHO) by ARM`.
//!
//! ## The kit (`nonmem_anchor/nested_levels_3_ar1/`)
//!
//! - `nested_levels_3_ar1.csv` (`make_data.py`) — `nested_levels_3.csv` plus `VIS`, a
//!   row's visit index inside its arm, and `T1..T5`, the arm's visit times (1, 2, 4, 8,
//!   12: the gaps 1, 2, 4, 4 are not the lags, so `car1` is not `ar1`). 12 sponsors × 2
//!   studies × 2 arms × 5 visits; `ARM` is 1, 2 inside each study, so the blocks are
//!   nested by key, and `ARMID` numbers the arms across the file.
//! - `*.ctl` (`gen_ctl.py`) — the PR2 anchor's `exact3` `$PRED` at its NONMEM optimum
//!   (`../nested_levels_3_eval.ctl`), with `L2 = ARMID` making `EPS(1..5)` one shared
//!   vector per arm (`$SIGMA 1 FIX` ×5) and `Y = IPRED + SD·Σₖ Lⱼₖ(ρ)·EPS(k)`, `L` the
//!   closed-form Cholesky factor of the arm's correlation matrix and `SD² = 0.255164` —
//!   the spelling of `nonmem_anchor/residual_correlation/`. `ρ = THETA(5) FIX`;
//!   `MAXEVAL=0`, FOCEI (`INTER`). Outputs in `results/`, from the `nonmem:7.6.0-anchor`
//!   image (amd64). The arm64 image gives the same OBJ to ≤ 2.5e-8.
//! - The ferx side reads `nested_levels_3_ar1_ferx.csv`: the same rows sorted by TIME
//!   within ID, so a sponsor's four arms interleave and a row's AR(1) lag inside its arm
//!   is not its distance in the file. (NONMEM needs each `L2` record contiguous, so its
//!   copy keeps the arm order.)
//!
//! The probes: `ar1` at ρ = 0.5 and −0.4 (a sign an implementation that squared or
//! clamped ρ gets wrong), `car1` at 0.6, and a ρ = 0 control (a *free* ρ at 0 on the ferx
//! side, so the line is kept and the dense path scores it). ferx evaluates each point with
//! `outer_maxiter = 0`: two engines scoring one parameter vector, its EBEs found on the
//! FD inner gradient, its marginal the dense-`R` FOCEI one.
//!
//! ## Measured
//!
//! | quantity | worst \|ferx − NONMEM\|, Linux aarch64 | macOS arm64 | bound |
//! |---|---|---|---|
//! | OFV, 4 probes | 5.658e-8 (`car1_p06`) | 6.029e-8 | `6e-7` (10.6× above) |
//! | η̂ and κ̂, 4 × 108 values | 1.676e-7 | 1.677e-7 | `2e-6` (11.9× above) |
//!
//! At the default `inner_tol` the gap is 1.1e-6 on `ar1_p05`: see [`INNER_TOL`].
//!
//! Fed column-keyed grouping (the level key cut to the bare `ARM` label, so arm 1 of
//! study 1 and arm 1 of study 2 share a block), the anchor dies before it reaches the
//! bound: the merged arms share their visit times, so `E_RESIDUAL_CORRELATION_TIES`
//! refuses the `ar1` fit (measured on #2102). Nesting by key is pinned numerically by
//! `a_level_name_in_by_groups_by_the_level_nested_by_key` (the `by ARM` / `by ARMID`
//! bit-equality).

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, EstimationMethod, FitOptions, FitResult};

const MODEL: &str = include_str!("../nonmem_anchor/nested_levels_3.ferx");
const DATA: &str = "nonmem_anchor/nested_levels_3_ar1/nested_levels_3_ar1_ferx.csv";

/// `(probe, kind, ρ, NONMEM OBJ)` — the `-1000000000` row of
/// `nonmem_anchor/nested_levels_3_ar1/results/<probe>.ext` (NONMEM 7.6.0 anchor image,
/// `MAXEVAL=0`).
const PROBES: &[(&str, &str, f64, f64)] = &[
    ("ar1_p05", "ar1", 0.5, 151.628_723_511_136_37),
    ("ar1_m04", "ar1", -0.4, 139.693_239_819_744_49),
    ("car1_p06", "car1", 0.6, 112.160_179_019_337_11),
    ("ar1_r0", "ar1", 0.0, 107.043_571_062_000_17),
];

/// Worst realised `|ferx − NONMEM|` over [`PROBES`]: 5.658e-8 on Linux (`car1_p06`;
/// 6.029e-8 on macOS arm64). `6e-7` is 10.6× above it.
const OFV_TOL: f64 = 6e-7;
/// The EBE search's stopping tolerance. The line moves every subject's EBE search onto the
/// FD inner gradient, which at the default `inner_tol = 1e-5` stops early enough to leave
/// 1.097e-6 on the `ar1_p05` OFV (measured, macOS arm64): ferx's own value then moves by
/// 1.1e-6 between `inner_tol` 1e-5 and 1e-8 and not at all below 1e-8, so the residual is
/// the stopping point, not the likelihood. 1e-8 is where the objective stops moving.
const INNER_TOL: f64 = 1e-8;
/// Worst realised `|Δ|` over every η̂ and κ̂ against the full-precision `.phi`: 1.676e-7
/// on Linux (1.677e-7 on macOS arm64). `2e-6` is 11.9× above it.
const EBE_TOL: f64 = 2e-6;

/// The twin: `exact3` with `theta RHO` and the correlation line. ρ = 0 is written free.
fn twin(kind: &str, rho: f64) -> String {
    let lo = if kind == "car1" { "0.0" } else { "-0.99" };
    let theta = if rho == 0.0 {
        format!("  theta RHO(0.0, {lo}, 0.99)\n")
    } else {
        format!("  theta RHO({rho}, {lo}, 0.99) FIX\n")
    };
    MODEL
        .replace(
            "  sigma ADD ~ 0.255164\n",
            &format!("  sigma ADD ~ 0.255164\n{theta}"),
        )
        .replace(
            "  DV ~ additive(ADD)\n",
            &format!("  DV ~ additive(ADD)\n  correlation {kind}(RHO) by ARM\n"),
        )
}

fn eval(src: &str) -> FitResult {
    let parsed = parse_full_model(src).expect("twin parses");
    assert!(parsed.model.residual_serial.is_some(), "the line is kept");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        DATA,
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("anchor data loads");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: false,
        inner_tol: INNER_TOL,
        ..parsed.fit_options.clone()
    };
    fit(&parsed.model, &pop, &parsed.model.default_params, &opts).expect("anchor evaluation")
}

/// One `.phi` row: the subject id and `ETA(1)..ETA(7)`.
fn nonmem_phi(probe: &str) -> Vec<(u32, [f64; 7])> {
    let path = format!("nonmem_anchor/nested_levels_3_ar1/results/{probe}.phi");
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
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

/// Per subject, the distinct `(STUDY, SLOT)` pairs in ferx's file's first-row order — the
/// cells ferx reports κ̂ for. NONMEM's study slot is `ETA(1 + STUDY)`, its arm slot
/// `ETA(3 + SLOT)`.
fn cells_per_subject() -> Vec<(u32, Vec<(usize, usize)>)> {
    let text = std::fs::read_to_string(DATA).unwrap();
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

/// V7: ferx's OFV and every EBE against NONMEM at each probe, and the anchor fed its own
/// failure — the ρ = 0 evaluation scored against each correlated probe misses by orders
/// of magnitude more than the bound, so the bound tells a correlated `R` from none.
#[test]
fn nested_levels_with_a_residual_correlation_match_nonmem() {
    // The interleave the module docs claim: a sponsor's first rows are four arms.
    let first: Vec<String> = std::fs::read_to_string(DATA)
        .unwrap()
        .lines()
        .skip(1)
        .take(4)
        .map(|l| l.split(',').nth(3).unwrap().to_string())
        .collect();
    assert_eq!(
        first,
        ["111", "112", "121", "122"],
        "arms interleave in TIME order"
    );

    let cells = cells_per_subject();
    let mut worst_ofv = 0.0f64;
    let mut worst_ebe = 0.0f64;
    let mut control = f64::NAN;
    for &(probe, kind, rho, nm_ofv) in PROBES {
        let r = eval(&twin(kind, rho));
        let d = (r.ofv - nm_ofv).abs();
        eprintln!(
            "{probe}: ferx {:.14}  NONMEM {nm_ofv:.14}  |Δ| {d:.3e}",
            r.ofv
        );
        assert!(
            r.ofv.is_finite() && d < OFV_TOL,
            "{probe}: OFV {} vs {nm_ofv}",
            r.ofv
        );
        worst_ofv = worst_ofv.max(d);
        if rho == 0.0 {
            control = r.ofv;
        }

        let phi = nonmem_phi(probe);
        assert_eq!(phi.len(), 12);
        assert_eq!(r.subjects.len(), 12);
        let mut compared = 0usize;
        for (i, ((id, nm), (cid, cell))) in phi.iter().zip(&cells).enumerate() {
            assert_eq!(id, cid, "subject order");
            assert_eq!(cell.len(), 4, "subject {id}: cells {cell:?}");
            let kappas = &r.ebe_kappas[i];
            assert_eq!(kappas.len(), 4, "subject {id}: one κ̂ vector per cell");
            let mut check = |got: f64, want: f64, what: String| {
                assert!(got.is_finite(), "{probe} {what}: ferx {got}");
                let d = (got - want).abs();
                assert!(d < EBE_TOL, "{probe} {what}: ferx {got} vs NONMEM {want}");
                worst_ebe = worst_ebe.max(d);
                compared += 1;
            };
            check(r.subjects[i].eta[0], nm[0], format!("subject {id} η"));
            for (g, &(study, slot)) in cell.iter().enumerate() {
                check(
                    kappas[g][0],
                    nm[study],
                    format!("subject {id} κ_study {study}"),
                );
                check(
                    kappas[g][1],
                    nm[2 + slot],
                    format!("subject {id} κ_arm {slot}"),
                );
            }
        }
        assert_eq!(compared, 12 * 9, "{probe}");
    }
    eprintln!("worst |ΔOFV| {worst_ofv:.3e}, worst |ΔEBE| {worst_ebe:.3e}");

    // Fed its own failure: the control (no correlation in effect) against each correlated
    // probe. The nearest is `car1_p06` at |Δ| ≈ 5.1, 8.5e6 × the bound.
    assert!(control.is_finite());
    for &(probe, _, rho, nm_ofv) in PROBES {
        if rho != 0.0 {
            assert!(
                (control - nm_ofv).abs() > 1e3 * OFV_TOL,
                "{probe}: the anchor cannot tell ρ = {rho} from ρ = 0"
            );
        }
    }
}
