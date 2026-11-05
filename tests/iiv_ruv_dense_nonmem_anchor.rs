//! NONMEM anchor for `iiv_on_ruv` next to a dense residual covariance (#2656).
//!
//! ## The object
//!
//! One η scales every residual SD of a subject, so it scales the subject's whole `R` and
//! leaves the correlation matrix alone: `R_i(η) = exp(2η_r)·R_i(0)`. The rows of `R_i(0)`
//! are coupled either by a `block_sigma` across a total / unbound pair or by a `correlation`
//! line over the three rows of a block.
//!
//! ## The kit (`nonmem_anchor/iiv_ruv_dense/`)
//!
//! `make_data.py` (stdlib) writes the data and every control stream: one-compartment IV,
//! 30 subjects, two doses, η on CL and V, proportional error so `∂R/∂f` is live, and a
//! residual factor whose log has variance 0.25. Every side of the object is non-degenerate
//! at every probe: ρ ≠ 0, ω_RUV ≠ 0, and |η̂_ruv| reaches 1.0–1.3.
//!
//! - `bs_*.ctl` — `$SIGMA BLOCK(2)` with `L2`, `Y = F + F*EXP(ETA(3))*RES`.
//! - `cs_*.ctl`, `ar1_*.ctl` — `L2` shares `EPS(1..3)` across a block and each row mixes them
//!   through the closed-form Cholesky factor of `C(ρ)`.
//! - `bs_est.ctl`, `cs_est.ctl` — full `METHOD=1 INTER` estimations.
//!
//! ## Measured
//!
//! Linux arm64 and macOS arm64 print the same digits. NONMEM 7.5.1 (`results/*.ext`, `.tab`, `.res`):
//!
//! | probe | structure | ρ | ω_RUV | \|ΔOFV\| | \|Δη̂\| | \|ΔCWRES\| | max \|η̂_ruv\| |
//! |---|---|---|---|---|---|---|---|
//! | `bs_p06` | `block_sigma` | 0.6 | 0.25 | 4.4e-7 | 7.0e-8 | 1.5e-7 | 1.17 |
//! | `bs_m04` | `block_sigma` | −0.4 | 0.25 | 1.7e-6 | 1.3e-7 | 2.2e-7 | 1.04 |
//! | `bs_lo` | `block_sigma` | 0.6 | 0.05 | 4.6e-7 | 1.3e-7 | 1.6e-7 | 0.72 |
//! | `cs_p05` | `cs` | 0.5 | 0.25 | 1.4e-7 | 9.4e-8 | 1.8e-7 | 1.06 |
//! | `cs_m02` | `cs` | −0.2 | 0.25 | 3.1e-7 | 1.3e-7 | 1.8e-7 | 1.26 |
//! | `ar1_p06` | `ar1` | 0.6 | 0.25 | 2.2e-6 | 2.3e-7 | 3.1e-7 | 1.04 |
//! | `ar1_m04` | `ar1` | −0.4 | 0.25 | 3.3e-7 | 4.7e-8 | 7.2e-8 | 1.32 |
//!
//! Each term is worth whole units: at `bs_p06` (OFV 388.66) ρ = 0 scores 453.55 and no
//! `iiv_on_ruv` 509.24; at `cs_p05` (282.02), 365.42 and 482.25. Full fits from the control
//! streams' starts land 6.0e-7 (`bs_est`, 381.1738) and 4.3e-7 (`cs_est`, 267.0717) from
//! NONMEM's OBJ. Both loops run analytic at every probe, asserted below.
//!
//! ## NPDE is not anchored on NONMEM here
//!
//! NONMEM's `$TABLE NPDE ESAMPLE=2000 WRESCHOL` misses the exact reference distribution by
//! rms 0.61 (`bs_p06`) and 0.43 (`cs_p05`), where a second 2000-draw ferx seed moves 0.15.
//! Adjudicated outside both engines: `npde_oracle.py` draws the model directly, 400,000
//! replicates per subject, and ferx lands on it at rms 0.11, the Monte-Carlo floor of its
//! own 2000 draws. The gap is not about the residual η: at ω_RUV = 1e-4 NONMEM's first row
//! still reads 0.11 against the oracle's −1.56. What it is was not chased; on
//! `npde_block_sigma_nonmem_anchor.rs`, with one η and no η on V, the two tools do agree.
//! So NPDE is pinned to the oracle, and the `.res` tables keep NONMEM's column as measured.

use ferx_core::api::read_population_for;
use ferx_core::build_info::GradientMethodKind;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, EstimationMethod, FitOptions, FitResult};

const PAIR: &str = include_str!("../nonmem_anchor/iiv_ruv_dense/pair.ferx");
const TRIPLE: &str = include_str!("../nonmem_anchor/iiv_ruv_dense/triple.ferx");
const DIR: &str = "nonmem_anchor/iiv_ruv_dense";

/// Worst realised |ΔOFV| 2.18e-6 (`ar1_p06`, Linux and macOS arm64): the two inner tolerances on η̂.
/// 4.6× headroom. Dropping either term costs tens of units.
const OFV_TOL: f64 = 1e-5;
/// Worst realised |Δη̂| 2.35e-7 (`ar1_p06`); ~8×.
const EBE_TOL: f64 = 2e-6;
/// Worst realised |ΔCWRES| 3.10e-7 (`ar1_p06`); ~10×.
const CWRES_TOL: f64 = 3e-6;
/// Realised rms 0.107 and 0.112 against the oracle, the Monte-Carlo floor of 2000 draws; a
/// draw with no residual factor sits at 0.226 and 0.220 from its own oracle's reference.
const NPDE_TOL: f64 = 0.16;
/// Realised 1.1e-13 on both structures: the same objective by two routes.
const REDUCTION_TOL: f64 = 1e-9;
/// Realised 6.0e-7 and 4.3e-7, Linux and macOS arm64; the optimizers' own stopping noise sets this.
const FIT_TOL: f64 = 1e-3;

fn swap(s: &str, from: &str, to: &str) -> String {
    assert!(s.contains(from), "{from}");
    s.replace(from, to)
}

/// The `block_sigma` twin at `rho` and `omega_ruv`.
fn pair(rho: f64, omega_ruv: f64) -> String {
    let cov = rho * 0.12 * 0.20;
    let s = swap(
        PAIR,
        "[0.0144, 0.0144, 0.04]",
        &format!("[0.0144, {cov}, 0.04]"),
    );
    swap(
        &s,
        "omega ETA_RUV ~ 0.25 FIX",
        &format!("omega ETA_RUV ~ {omega_ruv} FIX"),
    )
}

/// The `correlation` twin with `kind` at `rho`.
fn triple(kind: &str, rho: f64) -> String {
    let s = swap(
        TRIPLE,
        "theta RHO(0.5, -0.9, 0.99) FIX",
        &format!("theta RHO({rho}, -0.9, 0.99) FIX"),
    );
    swap(
        &s,
        "correlation cs(RHO)",
        &format!("correlation {kind}(RHO)"),
    )
}

/// The twin with its residual random effect taken out altogether.
fn without_ruv(src: &str) -> String {
    let s = swap(src, "  iiv_on_ruv = ETA_RUV\n", "");
    let line = s
        .lines()
        .find(|l| l.contains("omega ETA_RUV"))
        .expect("omega line")
        .to_string();
    swap(&s, &format!("{line}\n"), "")
}

/// The `cs` twin at a free ρ of 0: the line is kept and the dense path scores it (a `FIX` 0
/// is dropped by the parser).
fn triple_free_rho_at_zero() -> String {
    swap(
        &triple("cs", 0.0),
        "theta RHO(0, -0.9, 0.99) FIX",
        "theta RHO(0.0, -0.4, 0.99)",
    )
}

fn run(src: &str, data: &str) -> FitResult {
    run_with(src, data, |_| {})
}

fn run_with(src: &str, data: &str, edit: impl FnOnce(&mut FitOptions)) -> FitResult {
    let parsed = parse_full_model(src).expect("twin parses");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        &format!("{DIR}/{data}"),
        None,
        None,
        None,
        &parsed.column_map,
    )
    .expect("anchor data reads");
    let mut opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    edit(&mut opts);
    fit(&parsed.model, &pop, &parsed.model.default_params, &opts).expect("anchor evaluates")
}

/// NONMEM's `OBJ` of `probe`, the `-1000000000` row of its `.ext`.
fn nm_ofv(probe: &str) -> f64 {
    let text = std::fs::read_to_string(format!("{DIR}/results/{probe}.ext")).unwrap();
    text.lines()
        .find(|l| l.trim_start().starts_with("-1000000000"))
        .and_then(|l| l.split_whitespace().last())
        .and_then(|v| v.parse().ok())
        .expect("final row")
}

/// `(ID, [ETA1, ETA2, ETA3])` from `probe`'s `FIRSTONLY` table.
fn nm_ebes(probe: &str) -> Vec<(String, [f64; 3])> {
    let text = std::fs::read_to_string(format!("{DIR}/results/{probe}.tab")).unwrap();
    text.lines()
        .skip(2)
        .map(|l| {
            let f: Vec<f64> = l.split_whitespace().map(|v| v.parse().unwrap()).collect();
            (format!("{}", f[0] as u32), [f[2], f[3], f[4]])
        })
        .collect()
}

fn probes() -> Vec<(&'static str, String, &'static str)> {
    vec![
        ("bs_p06", pair(0.6, 0.25), "pair.csv"),
        ("bs_m04", pair(-0.4, 0.25), "pair.csv"),
        ("bs_lo", pair(0.6, 0.05), "pair.csv"),
        ("cs_p05", triple("cs", 0.5), "triple.csv"),
        ("cs_m02", triple("cs", -0.2), "triple.csv"),
        ("ar1_p06", triple("ar1", 0.6), "triple.csv"),
        ("ar1_m04", triple("ar1", -0.4), "triple.csv"),
    ]
}

/// `(IPRED, CWRES, NPDE)` of every observation row of `probe`'s residual table.
fn nm_rows(probe: &str) -> Vec<[f64; 3]> {
    let text = std::fs::read_to_string(format!("{DIR}/results/{probe}.res")).unwrap();
    text.lines()
        .skip(2)
        .map(|l| {
            let f: Vec<f64> = l.split_whitespace().map(|v| v.parse().unwrap()).collect();
            (f[3], [f[4], f[5], f[6]])
        })
        .filter(|&(mdv, _)| mdv == 0.0)
        .map(|(_, row)| row)
        .collect()
}

/// The OFV, each η̂ and each CWRES against NONMEM, the three things a dense `R` under
/// `exp(2η_r)` reaches: `ln|R|`, the interaction term, the EBE search and `R̃`.
#[test]
fn iiv_on_ruv_with_a_dense_r_matches_nonmem_at_fixed_parameters() {
    for (probe, src, data) in probes() {
        let r = run(&src, data);
        let nm = nm_ofv(probe);
        assert!(r.ofv.is_finite(), "{probe}: OFV {}", r.ofv);
        let d_ofv = (r.ofv - nm).abs();
        let ebes = nm_ebes(probe);
        assert_eq!(ebes.len(), r.subjects.len(), "{probe}");
        let (mut worst, mut reach) = (0.0_f64, 0.0_f64);
        for (i, (id, eta)) in ebes.iter().enumerate() {
            assert_eq!(&r.subjects[i].id, id, "{probe}: subject order");
            for (k, &w) in eta.iter().enumerate() {
                let g = r.subjects[i].eta[k];
                let d = (g - w).abs();
                assert!(d.is_finite(), "{probe} {id} ETA{}: {g}", k + 1);
                worst = worst.max(d);
            }
            let e = eta[2].abs();
            assert!(e.is_finite(), "{probe} {id}: NONMEM η̂_ruv {e}");
            reach = reach.max(e);
        }
        let rows = nm_rows(probe);
        let cwres: Vec<f64> = r.subjects.iter().flat_map(|s| s.cwres.clone()).collect();
        assert_eq!(cwres.len(), rows.len(), "{probe}");
        let mut worst_cwres = 0.0_f64;
        for (&got, row) in cwres.iter().zip(&rows) {
            let d = (got - row[1]).abs();
            assert!(d.is_finite(), "{probe}: CWRES {got}");
            worst_cwres = worst_cwres.max(d);
        }
        eprintln!(
            "{probe}: ferx {:.10} NONMEM {nm:.10} |ΔOFV| {d_ofv:.3e} |Δη̂| {worst:.3e} \
             |ΔCWRES| {worst_cwres:.3e} max|η̂_ruv| {reach:.3}",
            r.ofv
        );
        assert!(d_ofv < OFV_TOL, "{probe}: |ΔOFV| {d_ofv:.3e}");
        assert!(worst < EBE_TOL, "{probe}: worst |Δη̂| {worst:.3e}");
        assert!(
            worst_cwres < CWRES_TOL,
            "{probe}: |ΔCWRES| {worst_cwres:.3e}"
        );
        assert!(reach > 0.5, "{probe}: η̂_ruv must be live, reached {reach}");
        let analytic = GradientMethodKind::Analytic.as_str();
        assert_eq!(r.gradient_method_inner, analytic, "{probe}");
        assert_eq!(r.gradient_method_outer, analytic, "{probe}");
        assert!(
            !r.warnings.iter().any(|w| w.contains("finite-difference")),
            "{probe}: {:?}",
            r.warnings
        );
    }
}

/// Both terms are live at the anchor's point: taking either out moves the OFV by whole
/// units, far outside `OFV_TOL`, so a site that dropped one could not pass.
#[test]
fn each_term_is_worth_whole_ofv_units_at_the_anchor() {
    let nm = nm_ofv("bs_p06");
    let no_rho = run(&pair(0.0, 0.25), "pair.csv").ofv;
    let no_ruv = run(&without_ruv(&pair(0.6, 0.25)), "pair.csv").ofv;
    eprintln!("bs_p06: NONMEM {nm:.4}, ρ = 0 {no_rho:.4}, no iiv_on_ruv {no_ruv:.4}");
    assert!((no_rho - nm).abs() > 10.0, "ρ: {no_rho} vs {nm}");
    assert!((no_ruv - nm).abs() > 10.0, "iiv_on_ruv: {no_ruv} vs {nm}");

    let nm = nm_ofv("cs_p05");
    let no_rho = run(&triple_free_rho_at_zero(), "triple.csv").ofv;
    let no_ruv = run(&without_ruv(&triple("cs", 0.5)), "triple.csv").ofv;
    eprintln!("cs_p05: NONMEM {nm:.4}, ρ = 0 {no_rho:.4}, no iiv_on_ruv {no_ruv:.4}");
    assert!((no_rho - nm).abs() > 10.0, "ρ: {no_rho} vs {nm}");
    assert!((no_ruv - nm).abs() > 10.0, "iiv_on_ruv: {no_ruv} vs {nm}");
}

/// ρ = 0 recovers plain `iiv_on_ruv`: a free ρ at 0 keeps the line and scores the dense
/// path, a `FIX` 0 drops it and scores the diagonal one, and the two must agree.
#[test]
fn rho_zero_recovers_plain_iiv_on_ruv() {
    let dense = run(&triple_free_rho_at_zero(), "triple.csv").ofv;
    let plain = run(&triple("cs", 0.0), "triple.csv").ofv;
    let gap = (dense - plain).abs();
    eprintln!("correlation: dense {dense:.10} diagonal {plain:.10} |Δ| {gap:.3e}");
    assert!(
        dense.is_finite() && gap < REDUCTION_TOL,
        "{dense} vs {plain}"
    );

    // A `block_sigma` with a zero off-diagonal against two independent sigmas.
    let dense_src = swap(
        &pair(0.0, 0.25),
        "[0.0144, 0, 0.04] FIX",
        "[0.0144, 0.0, 0.04]",
    );
    let plain_src = swap(
        &pair(0.0, 0.25),
        "block_sigma (PROP_TOTAL, PROP_UNBOUND) = [0.0144, 0, 0.04] FIX",
        "sigma PROP_TOTAL ~ 0.0144 (variance) FIX\n  sigma PROP_UNBOUND ~ 0.04 (variance) FIX",
    );
    let dense = run(&dense_src, "pair.csv").ofv;
    let plain = run(&plain_src, "pair.csv").ofv;
    let gap = (dense - plain).abs();
    eprintln!("block_sigma: dense {dense:.10} diagonal {plain:.10} |Δ| {gap:.3e}");
    assert!(
        dense.is_finite() && gap < REDUCTION_TOL,
        "{dense} vs {plain}"
    );
}

/// ω_RUV → 0 recovers the plain correlated model. Expanding at ω = 0, with `q_i = rᵀR⁻¹r`
/// and `η̂_r ≈ ω(q_i − n_i)`, each subject's objective moves by `ω·(2n_i − (q_i − n_i)²)`: the
/// gap is first order in ω, so it must fall by the same factor as ω and vanish with it.
#[test]
fn omega_ruv_to_zero_recovers_the_plain_correlation() {
    for (name, data, src_at, plain) in [
        (
            "block_sigma",
            "pair.csv",
            pair as fn(f64, f64) -> String,
            without_ruv(&pair(0.6, 0.25)),
        ),
        (
            "correlation",
            "triple.csv",
            |_, om| {
                swap(
                    &triple("cs", 0.5),
                    "ETA_RUV ~ 0.25",
                    &format!("ETA_RUV ~ {om}"),
                )
            },
            without_ruv(&triple("cs", 0.5)),
        ),
    ] {
        let plain = run(&plain, data).ofv;
        let gap = |om: f64| {
            let ofv = run(&src_at(0.6, om), data).ofv;
            assert!(ofv.is_finite(), "{name}: {ofv}");
            (ofv - plain).abs()
        };
        let (coarse, fine) = (gap(1e-6), gap(1e-8));
        let rate = fine / coarse;
        eprintln!("{name}: |Δ| {coarse:.3e} at ω = 1e-6, {fine:.3e} at 1e-8, ratio {rate:.3e}");
        assert!(fine < 1e-3, "{name}: {fine:e}");
        assert!((0.5e-2..2e-2).contains(&rate), "{name}: ratio {rate:e}");
    }
}

/// `npde_oracle.py`'s scores for `name`, one per observation row.
fn oracle_npde(name: &str) -> Vec<f64> {
    let text = std::fs::read_to_string(format!("{DIR}/results/{name}.csv")).unwrap();
    text.lines()
        .skip(1)
        .map(|l| l.split(',').nth(1).unwrap().parse().unwrap())
        .collect()
}

fn rms_gap(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    (sum / a.len() as f64).sqrt()
}

/// NPDE on the dense draw with `exp(2η_r)`, against the exact reference distribution
/// (`npde_oracle.py`, 400,000 replicates). The oracle's no-factor scores are the control:
/// they sit further than `NPDE_TOL` from ferx, so a draw that lost the scale fails here.
#[test]
fn npde_reference_distribution_matches_the_oracle() {
    for (probe, src, data) in [
        ("bs_p06", pair(0.6, 0.25), "pair.csv"),
        ("cs_p05", triple("cs", 0.5), "triple.csv"),
    ] {
        let r = run_with(&src, data, |o| {
            o.npde_nsim = 2000;
            o.npde_seed = Some(2656);
        });
        let got: Vec<f64> = r.subjects.iter().flat_map(|s| s.npde.clone()).collect();
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{probe}: NPDE not finite"
        );
        let want = oracle_npde(&format!("npde_{probe}"));
        let control = oracle_npde(&format!("npde_{probe}_noruv"));
        let (on, off) = (rms_gap(&got, &want), rms_gap(&got, &control));
        let nm: Vec<f64> = nm_rows(probe).iter().map(|row| row[2]).collect();
        eprintln!(
            "{probe}: rms ΔNPDE vs oracle {on:.4}, vs no-factor oracle {off:.4}, NONMEM vs \
             oracle {:.4}",
            rms_gap(&nm, &want)
        );
        assert!(on < NPDE_TOL, "{probe}: {on}");
        assert!(off > NPDE_TOL, "{probe}: the control passes at {off}");
    }
}

/// Full FOCEI fits from the control streams' starts reach NONMEM's objective.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn fits_reach_nonmem_objective() {
    let free = |s: &str| s.replace(" FIX", "");
    let pair_src = swap(&free(PAIR), "[0.0144, 0.0144, 0.04]", "[0.02, 0.005, 0.03]")
        .replace("TVCL(1.05,", "TVCL(1.2,")
        .replace("TVV(9.5,", "TVV(12.0,")
        .replace("~ 0.08", "~ 0.1")
        .replace("~ 0.05", "~ 0.1")
        .replace("~ 0.25", "~ 0.1");
    let triple_src = free(TRIPLE)
        .replace("TVCL(1.05,", "TVCL(1.2,")
        .replace("TVV(9.5,", "TVV(12.0,")
        .replace("RHO(0.5, -0.9, 0.99)", "RHO(0.3, -0.4, 0.95)")
        .replace("PROP_ERR ~ 0.16", "PROP_ERR ~ 0.2")
        .replace("~ 0.08", "~ 0.1")
        .replace("~ 0.05", "~ 0.1")
        .replace("~ 0.25", "~ 0.1");
    for (probe, src, data) in [
        ("bs_est", pair_src, "pair.csv"),
        ("cs_est", triple_src, "triple.csv"),
    ] {
        let r = run_with(&src, data, |o| o.outer_maxiter = 500);
        let nm = nm_ofv(probe);
        eprintln!("{probe}: ferx {:.8} NONMEM {nm:.8}", r.ofv);
        assert!(
            r.ofv.is_finite() && (r.ofv - nm).abs() < FIT_TOL,
            "{probe}: {}",
            r.ofv
        );
    }
}
