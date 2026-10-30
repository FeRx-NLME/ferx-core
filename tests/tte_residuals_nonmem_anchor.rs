//! Per-record time-to-event residuals at the EBEs against NONMEM and an exact identity
//! (#901, tests T10–T12). No new NONMEM run: the references are the #1885 tables.
//!
//! * **T10** — `nonmem_anchor/tte_exponential_ebe.ctl` (analytic exponential hazard,
//!   η on log λ, 30 subjects, one record each). Two checks: ferx's `MARTINGALE` against
//!   the exact EBE identity `M = η̂/ω`, and against NONMEM's `DV − CHZ` at its POSTHOC.
//! * **T11** — `nonmem_anchor/pktte_ebe.ctl` (joint PK-TTE, η on CL, the hazard a
//!   function of the concentration): `MARTINGALE` against NONMEM's `DV − CHZ` at the TTE
//!   record. Every subject is *expected* to be scored off the exact-derivative walk
//!   (#2482: no SS dose, and every record time after the first dose), but the scope
//!   predicate is crate-private and this file does not assert it. It would not matter if
//!   it did: this anchor cannot see an engine swap — the walk, the #570 share and the
//!   dedicated solve agree to solver tolerance, inside the bound — which is why the
//!   `tte_record_terms_*` unit tests in `stats::likelihood` pin each arm bit for bit.
//! * **T12** — clock-forward and clock-reset RTTE, exponential, η on log λ, on
//!   `data/rtte_exponential.csv`: the per-subject sum `Σₖ Mᵢₖ = η̂ᵢ/ω`, which only holds
//!   when each record carries its own increment (or gap) and not `H(tₖ)`.
//!
//! **Why the identity.** For a hazard with η on log λ, every scored increment is `∝ e^η`,
//! so the inner objective `Σₖ [ΔHₖ − δₖ(log λ + η)] + η²/2ω` is stationary exactly where
//! `Σₖ (δₖ − ΔHₖ) = η/ω`. It is exact at the mode, so it tests η̂ *and* `ΔH` together —
//! and it is per subject: `Σᵢ Mᵢ` is **not** ≈ 0 at the population optimum (1.716 on
//! T10's data in NONMEM's own table), so that is not asserted.
//!
//! Every fit here is `outer_maxiter = 0` with everything at a stated point, so η̂ is the
//! inner problem alone. Since #2482 the TTE inner gradient is exact (none of these fits
//! warns of finite-difference inner gradients), which is why the identity now holds to
//! ~1e-11 rather than the ~1e-9 it did on FD gradients.

#![cfg(feature = "survival")]

use ferx_core::api::read_population_for;
use ferx_core::fit;
use ferx_core::io::output::sdtab;
use ferx_core::parser::model_parser::{parse_full_model, parse_model_string};
use ferx_core::types::{FitOptions, FitResult, OmegaMatrix, Population};

/// One `MARTINGALE` per TTE row of `r`, as `(id, time, M)`, cross-checked against the
/// sdtab the CLI writes: the TTE block's `MARTINGALE` cells must be these values.
fn martingales(r: &FitResult, pop: &Population) -> Vec<(String, f64, f64)> {
    let rows: Vec<(String, f64, f64)> = r
        .subjects
        .iter()
        .flat_map(|s| {
            s.tte_rows
                .iter()
                .map(move |t| (s.id.clone(), t.time, t.martingale))
        })
        .collect();
    let cols = sdtab(r, pop);
    let col = |n: &str| &cols.iter().find(|(c, _)| c == n).expect(n).1;
    let m = col("MARTINGALE");
    let tail = &m[m.len() - rows.len()..];
    for ((_, _, want), got) in rows.iter().zip(tail) {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "sdtab MARTINGALE vs tte_rows"
        );
    }
    assert!(
        m[..m.len() - rows.len()].iter().all(|x| x.is_nan()),
        "no MARTINGALE on a non-TTE row"
    );
    rows
}

/// Worst `|a − b|` over pairs, refusing a non-finite value first: `f64::max` drops NaN.
fn worst_abs(pairs: impl IntoIterator<Item = (f64, f64)>) -> f64 {
    let mut w = 0.0f64;
    for (a, b) in pairs {
        assert!(a.is_finite() && b.is_finite(), "non-finite: {a} vs {b}");
        w = w.max((a - b).abs());
    }
    w
}

// ── T10 ────────────────────────────────────────────────────────────────────────────

const OMEGA_T10: f64 = 2.84734e-1;

/// `(id, time, dv, chz, eta1)` per subject of `tte_exponential_ebe.tab`.
fn t10_nonmem() -> Vec<(String, f64, f64, f64, f64)> {
    std::fs::read_to_string("nonmem_anchor/results/tte_exponential_ebe.tab")
        .expect("NONMEM table")
        .lines()
        .skip(2)
        .filter_map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            // ID TIME DV CHZ ETA1
            (f.len() >= 5).then(|| (format!("{}", f[0] as i64), f[1], f[2], f[3], f[4]))
        })
        .collect()
}

#[test]
fn t10_martingale_is_eta_over_omega_and_matches_nonmem() {
    let src = std::fs::read_to_string("examples/tte_exponential.ferx").expect("model file");
    let m = parse_model_string(&src).expect("parse");
    let (pop, _) =
        read_population_for(&m, &None, "data/tte_exponential.csv", None, None, None, &[])
            .expect("data");
    let mut p = m.default_params.clone();
    p.theta[0] = 3.32303e-2;
    p.omega = OmegaMatrix::from_diagonal(&[OMEGA_T10], vec!["ETA_LAMBDA".into()]);
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        inner_tol: 1e-10,
        verbose: false,
        ..FitOptions::default()
    };
    let r = fit(&m, &pop, &p, &opts).expect("inner-only fit");
    let ms = martingales(&r, &pop);
    assert_eq!(ms.len(), 30, "one TTE row per subject");

    // (a) the exact identity, at ferx's own η̂.
    let wa = worst_abs(r.subjects.iter().map(|s| {
        let (_, _, mi) = ms.iter().find(|(i, ..)| *i == s.id).expect("row");
        (*mi, s.eta[0] / OMEGA_T10)
    }));
    eprintln!("T10(a) worst |M − η̂/ω| {wa:.3e}");

    // (b) NONMEM's `DV − CHZ` at its POSTHOC.
    let nm = t10_nonmem();
    assert_eq!(nm.len(), 30);
    let wb = worst_abs(nm.iter().map(|(id, t, dv, chz, _)| {
        let (_, _, mi) = ms
            .iter()
            .find(|(i, tt, _)| i == id && tt == t)
            .unwrap_or_else(|| panic!("subject {id} at {t}"));
        (*mi, dv - chz)
    }));
    eprintln!("T10(b) worst |M − (DV − CHZ)| {wb:.3e}");
    // Non-degeneracy: the residuals are not all one value, and the η's are live.
    assert!(nm.iter().all(|r| r.4.abs() > 1e-3));
    assert!(ms.iter().any(|m| m.2 > 0.0) && ms.iter().any(|m| m.2 < 0.0));
    assert!(wa < BOUND_T10A, "(a) {wa:.3e}");
    assert!(wb < BOUND_T10B, "(b) {wb:.3e}");
}

/// Measured 8.945e-12 on Linux aarch64 (`tools/linux-test.sh`) and macOS arm64 alike, at
/// `inner_tol = 1e-10`: the inner optimizer's stopping point, since the identity is exact at
/// the mode. ~11x headroom. (2.366e-9 before #2482 made the TTE inner gradient exact; a
/// bound left at that era's 2.5e-8 would have been ~2800x loose.)
const BOUND_T10A: f64 = 1e-10;
/// Measured 5.474e-9, Linux aarch64 and macOS arm64 alike: NONMEM's η̂ and the table's
/// 9-significant-figure print, unchanged by #2482. ~9x headroom.
const BOUND_T10B: f64 = 5e-8;

// ── T11 ────────────────────────────────────────────────────────────────────────────

/// `(id, time, dv − chz)` at each subject's TTE record (CMT 3, EVID 0) of `pktte_ebe.tab`.
fn t11_nonmem() -> Vec<(String, f64, f64)> {
    std::fs::read_to_string("nonmem_anchor/results/pktte_ebe.tab")
        .expect("NONMEM table")
        .lines()
        .skip(2)
        .filter_map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            // ID TIME EVID CMT DV IPRED CHZ HAZ ETA1 MDV
            (f.len() >= 10 && f[2] == 0.0 && f[3] == 3.0)
                .then(|| (format!("{}", f[0] as i64), f[1], f[4] - f[6]))
        })
        .collect()
}

#[test]
fn t11_joint_pktte_martingale_matches_nonmem() {
    let src = std::fs::read_to_string("nonmem_anchor/pktte_ebe_fit.ferx").expect("model");
    let parsed = parse_full_model(&src).expect("parse");
    let m = &parsed.model;
    let (pop, _) = read_population_for(
        m,
        &None,
        "nonmem_anchor/pktte_ebe.csv",
        None,
        None,
        None,
        &[],
    )
    .expect("data");
    let opts = FitOptions {
        run_covariance_step: false,
        inner_tol: 1e-10,
        ..parsed.fit_options.clone()
    };
    let r = fit(m, &pop, &m.default_params, &opts).expect("inner-only fit");
    let ms = martingales(&r, &pop);
    let nm = t11_nonmem();
    assert_eq!((ms.len(), nm.len()), (8, 8), "one TTE record per subject");
    let w = worst_abs(nm.iter().map(|(id, t, want)| {
        let (_, _, mi) = ms
            .iter()
            .find(|(i, tt, _)| i == id && tt == t)
            .unwrap_or_else(|| panic!("subject {id} at {t}"));
        (*mi, *want)
    }));
    eprintln!("T11 worst |M − (DV − CHZ)| {w:.3e}");
    assert!(nm.iter().any(|r| r.2 > 0.0) && nm.iter().any(|r| r.2 < 0.0));
    assert!(w < BOUND_T11, "{w:.3e}");
}

/// Measured 3.901e-9 on Linux aarch64 and macOS arm64 alike, scored off the #2482 walk
/// (3.655e-9 off the #570 share before it); the plan's 1e-8, 2.6x headroom. An engine swap
/// moves `M` by solver tolerance and passes this bound: that swap is the
/// `tte_record_terms_*` unit tests' to catch, not this one's.
const BOUND_T11: f64 = 1e-8;

// ── T12 ────────────────────────────────────────────────────────────────────────────

const OMEGA_T12: f64 = 0.09;

fn rtte_model(clock: &str) -> String {
    format!(
        "
[parameters]
  theta TVLAMBDA(0.15, 0.001, 10.0)
  omega ETA_LAMBDA ~ {OMEGA_T12}

[event_model]
  cmt    = 2
  type   = rtte
  clock  = {clock}
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)

[fit_options]
  method = focei
"
    )
}

#[test]
fn t12_rtte_subject_martingale_sum_is_eta_over_omega() {
    for clock in ["forward", "reset"] {
        let m = parse_model_string(&rtte_model(clock)).expect("parse");
        let (pop, _) = read_population_for(
            &m,
            &None,
            "data/rtte_exponential.csv",
            None,
            None,
            None,
            &[],
        )
        .expect("data");
        let opts = FitOptions {
            outer_maxiter: 0,
            run_covariance_step: false,
            inner_tol: 1e-10,
            ..FitOptions::default()
        };
        let r = fit(&m, &pop, &m.default_params, &opts).expect("inner-only fit");
        let ms = martingales(&r, &pop);
        let multi = r.subjects.iter().filter(|s| s.tte_rows.len() >= 3).count();
        assert!(
            multi > 10,
            "{clock}: the stream must have multi-record subjects"
        );
        let w = worst_abs(r.subjects.iter().map(|s| {
            let sum: f64 = ms.iter().filter(|(i, ..)| *i == s.id).map(|m| m.2).sum();
            (sum, s.eta[0] / OMEGA_T12)
        }));
        eprintln!(
            "T12 {clock}: {} subjects, worst |ΣM − η̂/ω| {w:.3e}",
            r.subjects.len()
        );
        assert!(w < BOUND_T12, "{clock}: {w:.3e}");
    }
}

/// Measured 8.430e-11 on both clocks, Linux aarch64 and macOS arm64 alike: the summed
/// stopping error of ~4 records per subject. ~12x headroom. (1.48e-8 before #2482's exact
/// inner gradient.)
const BOUND_T12: f64 = 1e-9;
