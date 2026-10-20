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
//! - `nested_levels_3_cov.ctl` (outputs in `results/`) — `nested_levels_3.ctl` with only
//!   `$COVARIANCE MATRIX=R SIGL=12 PRINT=E` and no `$TABLE` (#2177): the same OBJ to the
//!   bit, and the R-matrix SEs at a precision ferx's analytic R can be compared against.
//!   Stock `nmfe76` in `nonmem:7.6.0-anchor` (`nm3` rejects `$OMEGA SAME`).
//! - `nested_levels_3.ferx` — the ferx twin: two `level` lines and a weighted kappa at
//!   `ARM`, initial estimates set to the same values.
//!
//! ferx evaluates that point with `outer_maxiter = 0`, so the comparison is two engines
//! scoring one parameter vector, not two optimizers racing.
//!
//! ## Measured (macOS arm64)
//!
//! | quantity | worst \|ferx − NONMEM\|, analytic (PR3) | FD route (PR2) | bound |
//! |---|---|---|---|
//! | OFV | 1.121e-7 | 9.869e-8 | `1e-6` (8.9× above) |
//! | η̂ (sponsor), κ̂ (study, arm), 108 values | 8.821e-6 | 8.479e-6 | `2e-5` (2.3× above) |
//!
//! The engine under test is the analytic route (#1082 PR3): the compartment-free walk
//! seeds each κ on its compact axis, so the EBE search runs on the analytic inner gradient
//! and the outer gradient is analytic too (both asserted below, with no FD-fallback
//! warning). The κ columns of the FOCEI `H` stay central differences of the per-cell
//! prediction, as under one level. PR2's FD route scored the same point within the same
//! bounds, which is the comparison the second column records.

use ferx_core::api::read_population_for;
use ferx_core::build_info::GradientMethodKind;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, CovarianceMethod, CovarianceStatus, EstimationMethod, FitOptions, FitResult};
use std::path::Path;

/// NONMEM 7.6.0 `OBJ` at the evaluated point (`results/nested_levels_3_eval.ext`, `-1000000000`).
const NM_OFV: f64 = 107.04357106164440;
/// Worst realised |ΔOFV| 1.121e-7 on the analytic route (macOS arm64; 9.869e-8 on PR2's
/// FD route); `1e-6` is 8.9× above it. A one-level misreading of the data (κ_study drawn
/// per arm) moves the OFV by whole units.
const OFV_TOL: f64 = 1e-6;
/// Worst realised |Δ| over every η̂ and κ̂ against `.phi`'s six-significant-figure
/// print: 8.821e-6 on the analytic route (macOS arm64; 8.479e-6 on FD), of which up to
/// 5e-6 is the print's own rounding on values up to 3.2; `2e-5` is 2.3× above it.
const EBE_TOL: f64 = 2e-5;

const MODEL: &str = include_str!("../nonmem_anchor/nested_levels_3.ferx");

fn eval(model_src: &str) -> FitResult {
    eval_with(model_src, false)
}

/// [`eval`], optionally with the covariance step: `MATRIX=R` asked for explicitly, on the
/// default (analytic) route.
fn eval_with(model_src: &str, covariance: bool) -> FitResult {
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
        run_covariance_step: covariance,
        covariance_method: CovarianceMethod::Hessian,
        covariance_method_set: covariance,
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

    // The route (#1082 PR3): the compartment-free walk serves the nested model on both
    // loops, so the EBEs above were found on the analytic inner gradient, and no subject
    // fell back to finite differences.
    assert_eq!(
        r.gradient_method_inner,
        GradientMethodKind::Analytic.as_str()
    );
    assert_eq!(
        r.gradient_method_outer,
        GradientMethodKind::Analytic.as_str()
    );
    assert!(
        !r.warnings.iter().any(|w| w.contains("finite-difference")),
        "{:?}",
        r.warnings
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
        "nested_levels_3_cov.ctl",
        "results/nested_levels_3_cov.ext",
        "results/nested_levels_3_cov.lst",
        "results/nested_levels_3_cov.cov",
    ] {
        assert!(Path::new("nonmem_anchor").join(f).exists(), "{f}");
    }
}

/// The covariance step on a nested model takes the exact analytic R-matrix (#486): the
/// adjoint records over the compact axes and places the prior per group of each level, so a
/// phase does not own its own occasion κ. Asking for the analytic matrix and for the stencil
/// therefore gives different bits (the route, asserted) and the same matrix to the stencil's
/// accuracy. Measured, scaled by the diagonals: 1.7e-3 at the default `fd_hessian_step`
/// (1e-2) and `inner_tol`, 1.5e-4 at step 3e-3 with `inner_tol` 1e-9 (used here), 7.0e-4 at
/// step 1e-3 (EBE noise over h² takes over) — the stencil converging on the analytic matrix,
/// whose own check is against Richardson FD (1.65e-5 on a nested subject). Bound 3× the worst.
///
/// The fixture is closed-form PK on purpose: a compartment-free model is outside the
/// analytic covariance scope for a reason of its own, so it could not see the gate.
/// `data/warfarin_iov.csv` gains a `PHASE` level inside each occasion (its dose and first
/// four samples, then the rest); three subjects keep the stencil small.
#[test]
fn the_covariance_step_finite_differences_a_nested_pk_model() {
    let csv = std::fs::read_to_string("data/warfarin_iov.csv").unwrap();
    let mut out = String::new();
    let mut seen: std::collections::HashMap<(String, String), usize> = Default::default();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            out.push_str(line);
            out.push_str(",PHASE\n");
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let n = seen
            .entry((f[0].to_string(), f[8].to_string()))
            .or_insert(0);
        *n += 1;
        out.push_str(&format!("{line},{}\n", if *n <= 5 { 1 } else { 2 }));
    }
    let path = std::env::temp_dir().join(format!("ferx_nested_pk_{}.csv", std::process::id()));
    std::fs::write(&path, out).unwrap();
    let src = "[parameters]\n  theta TVCL(0.13, 0.001, 10.0)\n  theta TVV(8.0, 0.1, 500.0)\n  \
               theta TVKA(1.0, 0.01, 50.0)\n  omega ETA_CL ~ 0.09\n  \
               level OCC by OCC in ID\n  level PHASE by PHASE in OCC\n  \
               kappa K_OCC ~ 0.02 at OCC\n  kappa K_PH ~ 0.01 at PHASE\n  \
               sigma PROP ~ 0.1 (sd)\n[individual_parameters]\n  \
               CL = TVCL * exp(ETA_CL + K_OCC)\n  V = TVV * exp(K_PH)\n  KA = TVKA\n\
               [structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n[error_model]\n  \
               DV ~ proportional(PROP)\n";
    let run = |analytic: bool| {
        let parsed = parse_full_model(src).unwrap();
        assert!(parsed.model.n_kappa == 2 && parsed.fit_options.iov_column.is_none());
        let (mut pop, _) = read_population_for(
            &parsed.model,
            &parsed.covariate_decls,
            path.to_str().unwrap(),
            None,
            None,
            None,
            &parsed.column_map,
        )
        .unwrap();
        pop.subjects.truncate(3);
        let opts = FitOptions {
            method: EstimationMethod::FoceI,
            outer_maxiter: 0,
            run_covariance_step: true,
            analytic_cov_hessian: analytic,
            fd_hessian_step: 3e-3,
            inner_tol: 1e-9,
            ..parsed.fit_options.clone()
        };
        fit(&parsed.model, &pop, &parsed.model.default_params, &opts).unwrap()
    };
    let (a, fd) = (run(true), run(false));
    let _ = std::fs::remove_file(&path);
    // Three subjects, two occasions, two phases each: four cells per subject.
    assert!(
        a.ebe_kappas.iter().all(|k| k.len() == 4),
        "{:?}",
        a.ebe_kappas
    );
    let cov = |r: &FitResult| r.covariance_matrix.clone().expect("a covariance matrix");
    assert!(cov(&fd).iter().all(|v| v.is_finite()), "{}", cov(&fd));
    let free = (0..cov(&fd).nrows())
        .filter(|&i| cov(&fd)[(i, i)] > 0.0)
        .count();
    assert_eq!(free, 7, "three θ, ω², two κ variances, σ: {}", cov(&fd));
    // The analytic route was taken: it cannot reproduce the stencil's bits.
    let bits = |m: nalgebra::DMatrix<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_ne!(
        bits(cov(&a)),
        bits(cov(&fd)),
        "the nested covariance stayed on the stencil"
    );
    let (ca, cf) = (cov(&a), cov(&fd));
    let mut worst = 0.0f64;
    for i in 0..ca.nrows() {
        for j in 0..ca.ncols() {
            let scale = (cf[(i, i)] * cf[(j, j)]).abs().sqrt();
            if scale > 0.0 {
                assert!(ca[(i, j)].is_finite(), "analytic {ca}");
                worst = worst.max((ca[(i, j)] - cf[(i, j)]).abs() / scale);
            }
        }
    }
    eprintln!("MEASURE nested covariance analytic vs stencil: {worst:.3e}");
    assert!(worst < 5e-4, "analytic {ca} vs fd {cf}: {worst:.3e}");
}

/// The NONMEM columns the eight free parameters' SEs sit in, in ferx's order: θ, ω², κ², σ².
/// `OMEGA(2,2)` and `OMEGA(4,4)` are the first slot of each `SAME` chain, i.e. K_ST and K_ARM.
const SE_COLUMNS: [&str; 8] = [
    "THETA1",
    "THETA2",
    "THETA3",
    "THETA4",
    "OMEGA(1,1)",
    "OMEGA(2,2)",
    "OMEGA(4,4)",
    "SIGMA(1,1)",
];
const SE_NAMES: [&str; 8] = [
    "TVE0", "TVEMAX", "ED50", "ET50", "ETA_SP", "K_ST", "K_ARM", "ADD",
];

/// The `-1000000001` (standard error) row of a NONMEM `.ext`, picked by column name.
fn nonmem_ses(ext: &str) -> [f64; 8] {
    let text = std::fs::read_to_string(Path::new("nonmem_anchor/results").join(ext)).unwrap();
    let header: Vec<&str> = text
        .lines()
        .find(|l| l.trim_start().starts_with("ITERATION"))
        .expect("an .ext header")
        .split_whitespace()
        .collect();
    let row: Vec<f64> = text
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .find(|f| f.first() == Some(&"-1000000001"))
        .expect("an SE row")
        .iter()
        .map(|v| v.parse().unwrap())
        .collect();
    SE_COLUMNS.map(|c| row[header.iter().position(|h| *h == c).expect(c)])
}

/// The names of the parameters whose ferx SE is not within `tol` of `nm`, relatively.
fn se_misses(ferx: &[f64; 8], nm: &[f64; 8], tol: f64) -> Vec<&'static str> {
    (0..8)
        .filter(|&i| !((ferx[i] / nm[i] - 1.0).abs() < tol))
        .map(|i| SE_NAMES[i])
        .collect()
}

/// Worst realised |ferx/NONMEM − 1| over the eight SEs against `$COV MATRIX=R SIGL=12`
/// (`results/nested_levels_3_cov.ext`): 1.02% (ED50; macOS arm64 and Linux aarch64 agree
/// in every printed digit). `3%` is 2.9× above it, and well below the 11–27% by which the
/// committed run's default-precision `$COV` is off its own `SIGL=12` result on four of the
/// eight. NONMEM is not exact either: re-estimated at `NSIG=4 SIGL=12` (a slightly different
/// optimum), its SEs move by up to 2.2% (ED50) from this run, so the band is 1.36× above the
/// oracle's own spread. ferx is within 1.23% of that run too. It is the second reference below,
/// not this band, that pins the route.
const SE_TOL: f64 = 0.03;

/// The same eight SEs from a central-difference Hessian of ferx's own OFV, built outside
/// `compute_covariance`. Each point is a fresh `outer_maxiter = 0` evaluation with EBEs
/// re-found. The parameters are on NONMEM's natural scale (θ, ω², κ², σ² as values in the
/// model file), with relative step 3e-3 and `cov = 2·H⁻¹`. Release build, macOS arm64.
///
/// Steps 1e-2 and 3e-3 agree within 3.2e-4 (ED50). At 1e-3 the EBE noise over h² takes over
/// (6.8e-4 on ETA_SP).
const EXT_FD_SE: [f64; 8] = [
    1.274414e-1,
    2.976518,
    5.862647,
    1.341867e-1,
    5.711978e-2,
    4.809719e-2,
    1.426637,
    2.621309e-2,
];
/// Worst realised |analytic/external − 1|: 1.84e-4 (K_ST). The whole-population FD stencil
/// misses by 5.0e-3 (TVE0) and 2.2e-3 (ED50). `1e-3` sits 5.4× above the analytic gap,
/// 3.1× above the oracle's step-to-step spread, and 2.2× below the stencil's ED50 miss.
const ROUTE_TOL: f64 = 1e-3;

/// The covariance step on the nested anchor against NONMEM's `MATRIX=R` (#2177).
///
/// The reference is `nested_levels_3_cov.ctl`: the committed run's `.ctl` with only
/// `$COVARIANCE MATRIX=R SIGL=12` (plus `PRINT=E`), which converges to the same OBJ,
/// 107.04357104523862, bit for bit — one variable changed, and that variable is the
/// precision of NONMEM's own finite-difference R. At its default the committed run's SEs
/// are 20% low on TVEMAX and 24–27% high on ADD and ETA_SP; at `SIGL=12` they agree with
/// ferx's exact analytic R within 1.02%, and an external central-difference Hessian of
/// ferx's own OFV (natural scale, EBEs re-found at every point) agrees with the analytic R
/// to 2.3e-4. So the gap was NONMEM's precision, not ferx's matrix.
///
/// The straddle asserts the reference matters: the same check against the committed
/// default-precision row must fail, on at least TVEMAX, ETA_SP and ADD.
///
/// The 3% band cannot tell the analytic R from ferx's own FD stencil. With the analytic
/// assembly switched off, the stencil lands within 0.80% of NONMEM and emits no warning. So a
/// second check, against [`EXT_FD_SE`] at [`ROUTE_TOL`], pins the analytic route.
#[test]
fn nested_levels_r_ses_match_nonmem_matrix_r_at_sigl12() {
    let r = eval_with(MODEL, true);
    let se = |v: &Option<Vec<f64>>| v.clone().expect("an SE vector");
    let (t, o, k, s) = (
        se(&r.se_theta),
        se(&r.se_omega),
        se(&r.se_kappa),
        se(&r.se_sigma),
    );
    assert_eq!(
        (t.len(), o.len(), k.len(), s.len()),
        (4, 1, 2, 1),
        "{t:?} {o:?} {k:?} {s:?}"
    );
    assert_eq!(r.sigma_names, ["ADD"]);
    // ferx carries σ as an SD; NONMEM's SIGMA(1,1) SE is on the variance: d(σ²) = 2σ dσ.
    let ferx = [
        t[0],
        t[1],
        t[2],
        t[3],
        o[0],
        k[0],
        k[1],
        2.0 * r.sigma[0] * s[0],
    ];
    assert!(ferx.iter().all(|v| v.is_finite() && *v > 0.0), "{ferx:?}");
    assert_eq!(r.covariance_status, CovarianceStatus::Computed);
    assert_eq!(r.covariance_method, Some(CovarianceMethod::Hessian));
    // Measured empty. An FD inner-gradient fallback, a non-PD salvage, an SIR fallback or
    // an eigenvalue-floor note would each land here, and each would mean the SEs below are
    // not the analytic R's.
    assert!(r.warnings.is_empty(), "{:#?}", r.warnings);

    let nm = nonmem_ses("nested_levels_3_cov.ext");
    let mut worst = 0.0f64;
    for i in 0..8 {
        let rel = ferx[i] / nm[i] - 1.0;
        eprintln!(
            "MEASURE SE {:6} ferx {:.6e} NONMEM {:.6e} rel {rel:+.4e}",
            SE_NAMES[i], ferx[i], nm[i]
        );
        worst = worst.max(rel.abs());
    }
    eprintln!("MEASURE worst SE rel vs $COV SIGL=12: {worst:.4e}");
    assert_eq!(
        se_misses(&ferx, &nm, SE_TOL),
        Vec::<&str>::new(),
        "ferx {ferx:?} vs NONMEM {nm:?}"
    );

    let route_worst = (0..8)
        .map(|i| (ferx[i] / EXT_FD_SE[i] - 1.0).abs())
        .fold(0.0f64, f64::max);
    eprintln!("MEASURE worst SE rel vs external FD Hessian: {route_worst:.4e}");
    assert_eq!(
        se_misses(&ferx, &EXT_FD_SE, ROUTE_TOL),
        Vec::<&str>::new(),
        "not the analytic R: ferx {ferx:?} vs external FD {EXT_FD_SE:?}"
    );

    let default_precision = se_misses(&ferx, &nonmem_ses("nested_levels_3.ext"), SE_TOL);
    for p in ["TVEMAX", "ETA_SP", "ADD"] {
        assert!(
            default_precision.contains(&p),
            "{p} agrees with the default-precision $COV too, so the reference no longer \
             discriminates: misses {default_precision:?}"
        );
    }
}
