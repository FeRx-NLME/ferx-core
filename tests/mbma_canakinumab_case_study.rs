//! External anchor for the **bounded-endpoint** MBMA path (#1913): the canakinumab /
//! rheumatoid-arthritis case study of Bracis C, Taneja A, Lyauk YK, Barcomb H, de la
//! Peña A, Cellière G, *Model-Based Meta-Analysis With MonolixSuite: A Tutorial for
//! Longitudinal Categorical and Continuous Data*, CPT:PSP 2026;15:e70158
//! (doi:10.1002/psp4.70158), Case Study 2, final model (run r03): 9 studies, 15 arms,
//! 140 longitudinal ACR20 responder proportions, four drugs.
//!
//! `tests/mbma_naproxen_case_study.rs` anchors the continuous-endpoint path (SE
//! weighting, σ fixed to 1). This one covers what that cannot: a weighted **logit**
//! readout (`weight = 1/sqrt(NARM)`, σ **estimated**), the sample-size-weighted
//! between-arm κ (`kappa … weight = NARM`) and SIR, all on real literature data. The
//! model is `tests/data/mbma_canakinumab/mbma_canakinumab.ferx`. It keeps the
//! published `clamp` before the logit, but the clamp never binds on these data, so
//! nothing here tests it (see [`TOL_PUB_OFV`]).
//!
//! ## The data are not in the repository
//!
//! The article is CC BY-NC-**ND** 4.0 and its supplement carries no licence of its
//! own; a ferx-layout CSV is an adaptation, which ND forbids distributing. So the
//! tests read a locally provisioned file:
//!
//! ```text
//! python3 tools/mbma_canakinumab_fetch.py      # → data/external/mbma_canakinumab.csv (git-ignored)
//! ```
//!
//! or the path in `FERX_CANAKINUMAB_CSV`. Without it each data-reading test prints
//! `SKIPPED: …` and returns — **a green skip is not a passing anchor**. With
//! `FERX_REQUIRE_EXTERNAL_DATA=1` a missing file panics instead. The nightly
//! `slow-tests.yml` core leg fetches the data and sets that flag, so that is where
//! this anchor actually runs; in per-PR CI it skips. `no_external_data_is_tracked` runs everywhere
//! and fails if anything under `data/external/` is ever committed.
//!
//! ## Three references, three claims
//!
//! 1. **Monolix, objective level** (`objective_at_the_published_estimates_…`). At
//!    the Table S5 estimates, ferx's objective equals the published linearised −2LL
//!    once the two data-only terms are put back: `n·ln 2π` (ferx reports the OFV
//!    without the constant, as NONMEM does) and `Σ ln N_j` (the Jacobian of scoring
//!    `DV = transY/√N` on the natural scale; `docs/model-file/error-model.qmd`,
//!    "Objective-function value"). One evaluation sees the whole model: the
//!    logit-normal Emax, both drug covariate models, the κ weight, the residual
//!    weight and the data derivation.
//! 2. **NONMEM FOCE-I, the optimum** (`fit_reproduces_…`). Monolix fitted by SAEM,
//!    which maximises the *exact* likelihood; ferx's FOCE-I minimises the linearised
//!    one, and on this dataset the two optima differ (γ is 1.19 under FOCE against
//!    2.00 under SAEM; at its own optimum ferx's objective is 1.82 below its value at
//!    the published point). So the like-for-like reference for the converged fit is
//!    NONMEM 7.6.0 FOCE-I on the same model, `nonmem_anchor/mbma_canakinumab.ctl`.
//!    The published Monolix estimates are printed beside ferx's for the reader, and
//!    asserted only as "not a better point under this objective".
//! 3. **Profile likelihood, the standard errors.** NONMEM's own `$COV MATRIX=R` is
//!    not usable as the SE reference here: it puts SE(T50_POP) at 0.110, and a
//!    NONMEM profile of that one parameter (refits with T50 fixed at θ̂ ± 0.22) gives
//!    ΔOFV 0.971 / 0.904 — an SE of 0.227. So the SEs are compared against NONMEM
//!    profiles instead: each parameter fixed at θ̂ ± SE/2 in turn, every other
//!    parameter re-estimated, the SE read off the symmetric second difference (see
//!    [`REF_SE`]; runs in `nonmem_anchor/README.md`, "Canakinumab MBMA"). That is
//!    the curvature of NONMEM's objective, computed outside ferx. γ², whose profile
//!    is too asymmetric to have a curvature, is compared with NONMEM's `$COV` SE.

use ferx_core::types::FitOptions;
use ferx_core::{fit, prepare_run, FitResult};
use std::path::{Path, PathBuf};

const MODEL: &str = "tests/data/mbma_canakinumab/mbma_canakinumab.ferx";
const DEFAULT_DATA: &str = "data/external/mbma_canakinumab.csv";

// ---- Data fingerprint ------------------------------------------------------------
//
// The fetch script pins the derived file's sha256 (ecc4a9fb…f738). The test has no
// hashing crate, so it checks the shape and per-column sums as read by ferx's own
// reader instead: Σ DV for the response, Σ NARM for the weights, and one sum per
// drug indicator. A Y-derived DV, a dropped arm, a changed arm size or a recoded drug
// each moves at least one of them.
const N_STUDIES: usize = 9;
const N_OBS: usize = 140;
/// `Σ NARM` and `Σ` of each drug indicator over the 140 observation rows.
const SUM_COVARIATES: [(&str, f64); 4] = [
    ("NARM", 27608.0),
    ("ABATA", 40.0),
    ("ADALI", 43.0),
    ("CANAKI", 4.0),
];
/// `Σ DV` over the derived file, `DV = transY/√Narm` — the published transform, not
/// `logit(Y/100)`, which differs by up to 5.5e-3 per row.
const SUM_DV: f64 = -28.777668972433098;
/// `Σ ln N_j` over the 140 observations: the Jacobian of the natural-scale scoring.
const SUM_LN_N: f64 = 703.0046841807477;
/// `n·ln 2π`, the constant ferx (like NONMEM) leaves out of the reported OFV.
const N_LN_2PI: f64 = 257.3027892973083;

// ---- Monolix r03, Bracis et al. Table S5 ------------------------------------------
//
// Values as estimated, SEs from the linearised FIM. ω and γ are SDs; γ is the SD of
// the *unweighted* between-arm effect, so an arm of N has SD γ/√N.
const PUB_EMAX_POP: f64 = 0.416792287940095;
const PUB_B_EMAX_ABA: f64 = 1.70145268631664;
const PUB_B_EMAX_ADA: f64 = 1.12495818363337;
const PUB_B_EMAX_CAN: f64 = 1.1044872988958;
const PUB_T50_POP: f64 = 4.07578159106024;
const PUB_B_T50_ADA: f64 = -0.754034038063179;
const PUB_OMEGA_SD: f64 = 0.253262589363336;
const PUB_GAMMA_SD: f64 = 2.0045350753687;
const PUB_A: f64 = 2.46628905099576;
/// The published −2LL, linearisation.
const PUB_M2LL_LIN: f64 = 684.263422;

// ---- NONMEM 7.6.0 FOCE-I (nonmem_anchor/results/mbma_canakinumab.ext) ---------------
const NM_OFV: f64 = -277.86235425800453;
const NM_THETA: [f64; 6] = [
    0.42551719936241678,
    1.5142037978083973,
    1.3168971074091260,
    1.1464410527615598,
    4.1205572542075855,
    -0.71141436059298790,
];
const NM_SIGMA_SD: f64 = 2.4603771283118130;
const NM_OMEGA_VAR: f64 = 0.11769542907483575;
const NM_KAPPA_VAR: f64 = 1.4048880306803866;

// ---- External SE references (nonmem_anchor/README.md, "Canakinumab MBMA") ---------
//
// Order: 6 θ, σ (SD), ω², γ². The first eight are NONMEM profile SEs: the parameter
// fixed at θ̂ ± δ, δ = SE_ferx/2, everything else re-estimated, and
// `SE = δ·√(2/(ΔOFV₋ + ΔOFV₊))` — the symmetric second difference, which cancels the
// profile's cubic term. As δ → 0 that curvature is exactly what `R⁻¹` estimates.
//
// γ² is the exception. Its profile is 3× asymmetric even at ±SE/2 (ΔOFV 0.468 /
// 0.159: a variance whose SE exceeds its estimate), so no quadratic is there to
// read an SE off. Its reference is NONMEM's `$COV MATRIX=R` SE instead. That matrix
// is NOT used for the others: it puts SE(T50_POP) at 0.110 against a profile 0.227.
const REF_SE: [f64; 9] = [
    0.033155654195096834,
    0.14365384808413972,
    0.13429282222057873,
    0.3733431525876614,
    0.2269977066669139,
    0.12008598284917657,
    0.1552576115788109,
    0.07519767618387053,
    1.8712538083567720,
];

// ---- Tolerances ------------------------------------------------------------------
//
// Measured on Linux/aarch64 (tools/linux-test.sh) at base afe7f52b; macOS arm64 gave
// the same value in every printed digit.

/// −2LL at the published point vs Monolix. Realised **3.889e-7**, which is Monolix's
/// own print precision (6 decimals, ±5e-7). The bound is 25× that. Measured model
/// mutations move it by 10.8 (κ weight `√NARM`) to 2.5e4 (residual weight `1/NARM`).
/// Two edits leave it unmoved, by construction: σ `FIX`ed (this evaluation is at
/// the published σ either way; the fit test catches it, ΔOFV 0.063 vs NONMEM) and
/// the clamp removed (it never binds on these data — at ferx's optimum ACR20 spans
/// 0.068–0.759 individually, 0.083–0.715 at η = 0 — so no test here can see it).
const TOL_PUB_OFV: f64 = 1e-5;
/// Converged OFV vs NONMEM FOCE-I. Realised **3.7e-8**; the bound is ~270× that,
/// still four orders below the 1.82 that separates FOCE's optimum from SAEM's.
const TOL_NM_OFV: f64 = 1e-5;
/// Worst relative error of θ, σ, ω², γ² vs NONMEM. Realised **1.19e-5** (ω², the
/// flattest direction); bound 8.4× that.
const TOL_NM_REL: f64 = 1e-4;
/// Worst `|ln(SE_ferx / SE_ref)|`. Realised **0.0135** (ω²; next B_EMAX_ADA 0.0124).
/// The references carry their own noise — several profile refits end
/// `TERMINATED` (rounding errors) a few 1e-3 of OFV above their true minimum — so the
/// bound is 2.2× the realised gap, not tighter. Reporting ω²'s SE on the SD scale
/// instead would read |ln| 0.38.
const TOL_SE_RATIO: f64 = 0.03;

/// The derived dataset, or `None` (with a printed `SKIPPED:` line) when it has not
/// been provisioned. Panics instead under `FERX_REQUIRE_EXTERNAL_DATA=1`.
fn data_path() -> Option<PathBuf> {
    locate_data(
        std::env::var_os("FERX_CANAKINUMAB_CSV").map(PathBuf::from),
        require_flag(std::env::var("FERX_REQUIRE_EXTERNAL_DATA").ok().as_deref()),
    )
}

/// The value of `FERX_REQUIRE_EXTERNAL_DATA` that turns a skip into a panic: exactly
/// `"1"`, which is what `slow-tests.yml` sets on the leg that fetches the data, while
/// the other leg gets `""`.
fn require_flag(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Both sides of the flag, as the workflow spells them: `"1"` requires the data,
/// and the other leg's `""`, an unset variable, and `"0"` do not.
#[test]
fn require_flag_is_exactly_one() {
    assert!(require_flag(Some("1")));
    for v in [Some(""), None, Some("0")] {
        assert!(!require_flag(v), "{v:?} must not require the data");
    }
}

/// [`data_path`] with the environment passed in, so the require-or-skip rule is
/// testable without mutating process-global state.
fn locate_data(override_path: Option<PathBuf>, require: bool) -> Option<PathBuf> {
    let path =
        override_path.unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_DATA));
    if path.is_file() {
        return Some(path);
    }
    let msg = format!(
        "canakinumab anchor: no data at {} — run `python3 tools/mbma_canakinumab_fetch.py` \
         or set FERX_CANAKINUMAB_CSV (the data cannot be committed; #1913)",
        path.display()
    );
    if require {
        panic!("FERX_REQUIRE_EXTERNAL_DATA=1 and {msg}");
    }
    eprintln!("SKIPPED: {msg}");
    None
}

/// Acceptance 4 of #1913: a job that provisions the data sets
/// `FERX_REQUIRE_EXTERNAL_DATA=1`, and then a missing file is a red test, never a
/// green skip.
#[test]
#[should_panic(expected = "FERX_REQUIRE_EXTERNAL_DATA=1 and canakinumab anchor: no data at")]
fn a_missing_file_panics_when_external_data_is_required() {
    locate_data(
        Some(PathBuf::from("/nonexistent/mbma_canakinumab.csv")),
        true,
    );
}

/// …and without the flag the same missing file is a skip (`None`), not a panic —
/// the other side of the same gate, so forcing it either way reddens one of the two.
#[test]
fn a_missing_file_skips_when_external_data_is_not_required() {
    assert_eq!(
        locate_data(
            Some(PathBuf::from("/nonexistent/mbma_canakinumab.csv")),
            false
        ),
        None
    );
}

fn model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL)
}

/// The shipped model with its initial values replaced by the Table S5 estimates,
/// written to a temporary file so it goes through the same `prepare_run` pipeline
/// as the shipped one. The name carries the process id AND a per-call counter: the
/// two data tests run on parallel threads of one process, and each removes its file
/// after reading it, so a shared name would let one test delete the other's model
/// between its write and its `prepare_run`.
fn published_model_file() -> PathBuf {
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let src = std::fs::read_to_string(model_path()).expect("read the model");
    let subs = [
        ("EMAX_POP(0.40,", format!("EMAX_POP({PUB_EMAX_POP},")),
        ("B_EMAX_ABA(1.5,", format!("B_EMAX_ABA({PUB_B_EMAX_ABA},")),
        ("B_EMAX_ADA(1.0,", format!("B_EMAX_ADA({PUB_B_EMAX_ADA},")),
        ("B_EMAX_CAN(1.0,", format!("B_EMAX_CAN({PUB_B_EMAX_CAN},")),
        ("T50_POP(4.0,", format!("T50_POP({PUB_T50_POP},")),
        ("B_T50_ADA(-0.5,", format!("B_T50_ADA({PUB_B_T50_ADA},")),
        (
            "ETA_BSV_EMAX ~ 0.3 (sd)",
            format!("ETA_BSV_EMAX ~ {PUB_OMEGA_SD} (sd)"),
        ),
        (
            "KAPPA_EMAX ~ 2.0 (sd)",
            format!("KAPPA_EMAX ~ {PUB_GAMMA_SD} (sd)"),
        ),
        ("ADD_ERR ~ 2.5 (sd)", format!("ADD_ERR ~ {PUB_A} (sd)")),
    ];
    let mut out = src.clone();
    for (from, to) in &subs {
        assert_eq!(
            out.matches(from).count(),
            1,
            "the shipped model no longer contains `{from}` exactly once"
        );
        out = out.replace(from, to);
    }
    let path = std::env::temp_dir().join(format!(
        "ferx_mbma_canakinumab_published_{}_{call}.ferx",
        std::process::id()
    ));
    std::fs::write(&path, out).expect("write the published-value model");
    path
}

/// Shape and checksum of the provisioned file, through ferx's own reader.
fn check_population(pop: &ferx_core::types::Population) {
    assert_eq!(pop.subjects.len(), N_STUDIES, "9 studies");
    let dv: Vec<f64> = pop
        .subjects
        .iter()
        .flat_map(|s| s.observations.iter().copied())
        .collect();
    assert_eq!(dv.len(), N_OBS, "140 observations");
    assert!(dv.iter().all(|v| v.is_finite()), "non-finite DV");
    let sum: f64 = dv.iter().sum();
    assert!(
        (sum - SUM_DV).abs() < 1e-9,
        "Σ DV = {sum:.12}, expected {SUM_DV:.12}: not the file tools/mbma_canakinumab_fetch.py \
         derives (sha256 ecc4a9fb…f738)"
    );
    for (name, want) in SUM_COVARIATES {
        let mut got = 0.0;
        for s in &pop.subjects {
            // `obs_covariates` is empty when nothing varies within the subject, and
            // then `covariates` stands for every observation (the documented fallback;
            // studies 6, 8, 9 are single-arm). Otherwise it must be one row per obs.
            let rows: Vec<&std::collections::HashMap<String, f64>> = if s.obs_covariates.is_empty()
            {
                vec![&s.covariates; s.observations.len()]
            } else {
                assert_eq!(
                    s.obs_covariates.len(),
                    s.observations.len(),
                    "subject {}: one covariate row per observation",
                    s.id
                );
                s.obs_covariates.iter().collect()
            };
            for row in rows {
                // No `unwrap_or(0.0)`: a column the reader dropped must fail here,
                // not sum to a plausible number.
                got += row
                    .get(name)
                    .unwrap_or_else(|| panic!("subject {}: no {name} covariate", s.id));
            }
        }
        assert!(
            (got - want).abs() < 1e-9,
            "Σ {name} = {got}, expected {want}: not the file tools/mbma_canakinumab_fetch.py \
             derives (sha256 ecc4a9fb…f738)"
        );
    }
}

/// Acceptance 5 of #1913: the licence forbids redistributing the data or anything
/// derived from it, so nothing under `data/external/` may ever be tracked, and no
/// `mbma_canakinumab*.csv` anywhere. The second pathspec exists because the NONMEM
/// twin reads the same file next to its `.ctl` in `nonmem_anchor/`, a directory where
/// committing `<name>.csv` beside `<name>.ctl` is the convention. Runs on every PR;
/// needs no data.
#[test]
fn no_external_data_is_tracked() {
    let out = std::process::Command::new("git")
        .args(["ls-files", "--", "data/external", "*mbma_canakinumab*.csv"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("git must be runnable for this check");
    assert!(
        out.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let tracked = String::from_utf8_lossy(&out.stdout);
    assert!(
        tracked.trim().is_empty(),
        "canakinumab data are tracked, and may not be (CC BY-NC-ND; #1913):\n{tracked}"
    );
}

/// Reference 1: at the published estimates, ferx's objective is the published
/// linearised −2LL less the two data-only terms. An evaluation, not a fit.
#[test]
fn objective_at_the_published_estimates_matches_monolix_linearised_m2ll() {
    let Some(data) = data_path() else { return };
    let model = published_model_file();
    let prep = prepare_run(model.to_str().unwrap(), Some(data.to_str().unwrap()))
        .expect("prepare the published-value model");
    let _ = std::fs::remove_file(&model);
    check_population(&prep.population);
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..prep.parsed.fit_options.clone()
    };
    let r = fit(
        &prep.parsed.model,
        &prep.population,
        &prep.init_params,
        &opts,
    )
    .expect("evaluate at the published estimates");
    assert!(
        r.ofv.is_finite(),
        "OFV at the published estimates: {}",
        r.ofv
    );
    let m2ll = r.ofv + SUM_LN_N + N_LN_2PI;
    let err = (m2ll - PUB_M2LL_LIN).abs();
    eprintln!(
        "published point: ferx OFV {:.6} → −2LL {m2ll:.6} vs Monolix {PUB_M2LL_LIN:.6}, |Δ| {err:.3e}",
        r.ofv
    );
    assert!(
        err < TOL_PUB_OFV,
        "−2LL at the published estimates: ferx {m2ll:.6} vs Monolix {PUB_M2LL_LIN} (|Δ| {err:.3e})"
    );
}

fn fit_shipped(data: &Path) -> (FitResult, f64) {
    let prep = prepare_run(model_path().to_str().unwrap(), Some(data.to_str().unwrap()))
        .expect("prepare the shipped model");
    check_population(&prep.population);
    let opts = FitOptions {
        verbose: false,
        run_covariance_step: true,
        sir: true,
        sir_samples: 1000,
        sir_resamples: 250,
        sir_seed: Some(1913),
        ..prep.parsed.fit_options.clone()
    };
    let r = fit(
        &prep.parsed.model,
        &prep.population,
        &prep.init_params,
        &opts,
    )
    .expect("fit the canakinumab model");

    // The same objective at the published point, for claim 2's "not a better point".
    let model = published_model_file();
    let pprep = prepare_run(model.to_str().unwrap(), Some(data.to_str().unwrap()))
        .expect("prepare the published-value model");
    let _ = std::fs::remove_file(&model);
    let popts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..pprep.parsed.fit_options.clone()
    };
    let at_pub = fit(
        &pprep.parsed.model,
        &pprep.population,
        &pprep.init_params,
        &popts,
    )
    .expect("evaluate at the published estimates")
    .ofv;
    (r, at_pub)
}

/// References 2 and 3, and SIR: one converged FOCE-I fit with the covariance step
/// and in-fit SIR, against NONMEM FOCE-I's optimum and NONMEM profile SEs.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow convergence fit against a published MBMA case study (#1913): opt in with --features slow-tests"
)]
fn fit_reproduces_nonmem_foce_optimum_profile_ses_and_runs_sir() {
    let Some(data) = data_path() else { return };
    let (r, ofv_at_pub) = fit_shipped(&data);
    assert!(r.converged, "the published model must converge");
    assert!(r.ofv.is_finite() && ofv_at_pub.is_finite());

    let iov = r.omega_iov.as_ref().expect("a kappa fit");
    assert_eq!(r.kappa_names, ["KAPPA_EMAX"]);
    let se_theta = r.se_theta.as_ref().expect("covariance step: se_theta");
    let se_omega = r.se_omega.as_ref().expect("covariance step: se_omega");
    let se_sigma = r.se_sigma.as_ref().expect("covariance step: se_sigma");
    let se_kappa = r.se_kappa.as_ref().expect("covariance step: se_kappa");

    // ---- Monolix, for the reader ----
    let pubs = [
        PUB_EMAX_POP,
        PUB_B_EMAX_ABA,
        PUB_B_EMAX_ADA,
        PUB_B_EMAX_CAN,
        PUB_T50_POP,
        PUB_B_T50_ADA,
    ];
    eprintln!(
        "{:<14} {:>12} {:>12} {:>12}",
        "", "ferx", "NONMEM", "Monolix"
    );
    for i in 0..6 {
        eprintln!(
            "{:<14} {:>12.6} {:>12.6} {:>12.6}",
            r.theta_names[i], r.theta[i], NM_THETA[i], pubs[i]
        );
    }
    eprintln!(
        "{:<14} {:>12.6} {:>12.6} {:>12.6}\n{:<14} {:>12.6} {:>12.6} {:>12.6}\n{:<14} {:>12.6} {:>12.6} {:>12.6}",
        "ω (SD)", r.omega[(0, 0)].sqrt(), NM_OMEGA_VAR.sqrt(), PUB_OMEGA_SD,
        "γ (SD)", iov[(0, 0)].sqrt(), NM_KAPPA_VAR.sqrt(), PUB_GAMMA_SD,
        "a (SD)", r.sigma[0], NM_SIGMA_SD, PUB_A,
    );
    eprintln!(
        "OFV: ferx {:.6}, NONMEM {NM_OFV:.6}, ferx at the published point {ofv_at_pub:.6}",
        r.ofv
    );
    // SAEM's optimum of the exact likelihood is not a better point of the FOCE
    // objective than FOCE's own optimum.
    assert!(
        r.ofv <= ofv_at_pub,
        "ferx's optimum {:.6} is worse than the published point {ofv_at_pub:.6}",
        r.ofv
    );

    // ---- NONMEM FOCE-I optimum ----
    let d_ofv = (r.ofv - NM_OFV).abs();
    eprintln!("OFV vs NONMEM: |Δ| {d_ofv:.3e}");
    assert!(
        d_ofv < TOL_NM_OFV,
        "OFV: ferx {:.8} vs NONMEM {NM_OFV:.8} (|Δ| {d_ofv:.3e})",
        r.ofv
    );
    let est = [
        r.theta[0],
        r.theta[1],
        r.theta[2],
        r.theta[3],
        r.theta[4],
        r.theta[5],
        r.sigma[0],
        r.omega[(0, 0)],
        iov[(0, 0)],
    ];
    let nm = [
        NM_THETA[0],
        NM_THETA[1],
        NM_THETA[2],
        NM_THETA[3],
        NM_THETA[4],
        NM_THETA[5],
        NM_SIGMA_SD,
        NM_OMEGA_VAR,
        NM_KAPPA_VAR,
    ];
    let names = [
        "EMAX_POP",
        "B_EMAX_ABA",
        "B_EMAX_ADA",
        "B_EMAX_CAN",
        "T50_POP",
        "B_T50_ADA",
        "ADD_ERR (SD)",
        "ω² BSV",
        "γ² KAPPA",
    ];
    let se = [
        se_theta[0],
        se_theta[1],
        se_theta[2],
        se_theta[3],
        se_theta[4],
        se_theta[5],
        se_sigma[0],
        se_omega[0],
        se_kappa[0],
    ];
    let mut worst_rel = 0.0_f64;
    let mut worst_ratio = 0.0_f64;
    for i in 0..9 {
        // is_finite before folding: f64::max discards NaN.
        assert!(
            est[i].is_finite() && se[i].is_finite() && se[i] > 0.0,
            "{}: estimate {} / SE {} not finite and positive",
            names[i],
            est[i],
            se[i]
        );
        let rel = (est[i] - nm[i]).abs() / nm[i].abs();
        let ratio = (se[i] / REF_SE[i]).ln().abs();
        eprintln!(
            "{:<14} ferx {:>12.8} NONMEM {:>12.8} rel {rel:.3e} | SE ferx {:.6} ref {:.6} |ln ratio| {ratio:.4}",
            names[i], est[i], nm[i], se[i], REF_SE[i]
        );
        worst_rel = worst_rel.max(rel);
        worst_ratio = worst_ratio.max(ratio);
    }
    eprintln!(
        "worst rel vs NONMEM {worst_rel:.3e}, worst |ln SE ratio| vs reference {worst_ratio:.4}"
    );
    assert!(
        worst_rel < TOL_NM_REL,
        "worst relative error vs NONMEM {worst_rel:.3e} ≥ {TOL_NM_REL:e}"
    );
    assert!(
        worst_ratio < TOL_SE_RATIO,
        "worst |ln(SE_ferx / SE_ref)| {worst_ratio:.4} ≥ {TOL_SE_RATIO}"
    );

    // ---- SIR: a smoke check, not an anchor ----
    // SIR ran, is not degenerate, and gives finite, ordered intervals with γ² inside
    // its own. That is all. A wrong proposal that still yields usable draws would
    // pass; there is no external SIR reference for this dataset. The interval
    // contents are pinned against the covariance step in `tests/sir_ci_kappa.rs`.
    let ess = r.sir_ess.expect("in-fit SIR ran: sir_ess");
    assert!(ess.is_finite() && ess > 10.0, "degenerate SIR, ESS {ess}");
    for (what, ci) in [
        ("θ", &r.sir_ci_theta),
        ("Ω", &r.sir_ci_omega),
        ("σ", &r.sir_ci_sigma),
        ("κ", &r.sir_ci_kappa),
    ] {
        let ci = ci
            .as_ref()
            .unwrap_or_else(|| panic!("SIR filled no {what} interval"));
        assert!(
            ci.iter()
                .all(|(lo, hi)| lo.is_finite() && hi.is_finite() && lo < hi),
            "{what}: SIR interval not finite and ordered: {ci:?}"
        );
    }
    let (lo, hi) = r.sir_ci_kappa.as_ref().unwrap()[0];
    let kappa = iov[(0, 0)];
    eprintln!("SIR: ESS {ess:.1}, γ² {kappa:.6} in [{lo:.6}, {hi:.6}]");
    assert!(
        lo < kappa && kappa < hi,
        "SIR κ interval [{lo}, {hi}] does not bracket γ² = {kappa}"
    );
}
