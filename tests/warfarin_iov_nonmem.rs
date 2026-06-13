//! NONMEM 7.5.1 FOCEI cross-check for the inter-occasion variability (IOV) path
//! — issue #101.
//!
//! Validates ferx's augmented IOV marginal (`foce_subject_nll_iov`) against a
//! NONMEM reference fit on the 10-subject warfarin IOV dataset
//! (`data/warfarin_iov.csv`, 2 occasions/subject): 1-cpt oral, proportional
//! error, IOV on CL. Model: `examples/warfarin_iov.ferx`.
//!
//! ## What this guards
//!
//! `iov_objective_matches_nonmem` — ferx's FOCEI objective, evaluated at
//! NONMEM's final MLE (all parameters FIXed), compared to NONMEM's
//! OFV-without-constant. ferx lands at ≈308.2 vs NONMEM's 308.83 — a **~0.6-unit
//! match**.
//!
//! ### History (issues #101 / #104 / #109)
//!
//! This started as a ≈37-unit gap under the old Option-A superposition, fell to
//! ≈17 once the continuous per-occasion-aware prediction (issue #104,
//! `pk::predict_iov`) made the prediction exact, and finally **closed to ~0.6**
//! when the FOCEI INTER marginal switched from the augmented Sheiner–Beal
//! linearised form to the Almquist 2015 Laplace form (commit `2de0bea`), which
//! aligned ferx's marginal with NONMEM's Laplace FOCEI. That closure resolved
//! issue #109, whose residual was diagnosed as exactly this Sheiner–Beal-vs-Laplace
//! cross-engine difference.
//!
//! The remaining ~0.6 is well within what NONMEM's own non-clean convergence on
//! this dataset can explain — it terminated on ROUNDING ERRORS (ERROR=134),
//! though OFV and estimates were stable across the last iterations.
//!
//! **The prediction is exact.** ferx's population PRED (η=κ=0) matches NONMEM's
//! PRED to 5 significant figures on every row of the dataset, *including the
//! occasion-2 carryover rows* (e.g. t=120.5: 6.1882; t=124: 11.761).
//!
//! The simultaneous cross-occasion event ordering (occasion-1 obs and
//! occasion-2 dose both at t=120) was investigated as a candidate for the old
//! residual: making the event sort occasion-aware there changes the OFV by only
//! ~0.3 units, so it was not pursued (an occasion-aware tie-break would also
//! need per-event occasion data for EVID=2 records to stay correct — see #107).
//!
//! This test is `#[ignore]`d (it needs the NONMEM-anchored fixture) and now
//! guards that the IOV marginal stays in agreement with NONMEM.
//!
//! ## Reproducing the NONMEM reference
//!
//! NONMEM 7.5.1, FOCEI (`METHOD=1 INTER`), from `tests/nonmem/warfarin_iov.ctl`
//! over `data/warfarin_iov.csv`. IOV on CL is coded with one ETA per occasion
//! sharing a single variance via `$OMEGA BLOCK(1) ... SAME`:
//!
//! ```text
//! $SUBROUTINES ADVAN2 TRANS2
//! $PK
//!   OCC1 = 0
//!   OCC2 = 0
//!   IF(OCC.EQ.1) OCC1 = 1
//!   IF(OCC.EQ.2) OCC2 = 1
//!   IOVCL = OCC1*ETA(4) + OCC2*ETA(5)
//!   CL = THETA(1)*EXP(ETA(1) + IOVCL)
//!   V  = THETA(2)*EXP(ETA(2))
//!   KA = THETA(3)*EXP(ETA(3))
//!   S2 = V
//! $ERROR
//!   Y = F*(1 + EPS(1))
//! $OMEGA 0.09 ; 0.04 ; 0.30        (ETA_CL, ETA_V, ETA_KA)
//! $OMEGA BLOCK(1) 0.01   ; occasion 1
//! $OMEGA BLOCK(1) SAME   ; occasion 2  -> IOV
//! $SIGMA 0.04
//! $ESTIMATION METHOD=1 INTER MAXEVAL=9999 NSIG=3 SIGL=9 NOABORT
//! ```
//!
//! Final estimates (run.ext, iteration 151), OBJECTIVE FUNCTION VALUE WITHOUT
//! CONSTANT = 308.8305 (WITH CONSTANT 713.1634 = 308.8305 + 220·ln(2π)).
//! NONMEM minimization TERMINATED on rounding errors (ERROR=134) but the OFV
//! and estimates are stable across the last iterations.
//!
//! ## Per-occasion individual-CL cross-check (issue #238)
//!
//! `iov_individual_cl_matches_nonmem` guards the post-fit individual-parameter
//! columns (`sdtab` `CL`), which must carry each observation's **occasion**
//! kappa. Issue #238 fixed a bug where those columns silently used `kappa = 0`,
//! so an IOV subject's CL was identical across occasions.
//!
//! The test has two parts:
//!
//! - **Internal consistency** (always runs when opted in, no NONMEM needed):
//!   with all parameters FIXed at NONMEM's MLE, `V`/`KA` (BSV only) are constant
//!   across each subject's observations, while `CL` is constant *within* an
//!   occasion and differs *between* occasions — proving the per-occasion kappa
//!   is applied. The pre-#238 code makes CL occasion-independent for every
//!   subject, which this part detects.
//! - **Cross-engine check** (runs only when the reference fixture is present):
//!   ferx's per-`(ID, OCC)` CL is compared to NONMEM's within `CL_REL_TOL`.
//!
//! ### Producing the NONMEM CL reference
//!
//! 1. Run `tests/nonmem/warfarin_iov.ctl` (NONMEM 7.5.1, FOCEI) — its `$TABLE`
//!    now emits `CL V KA` to `sdtab_iov`. The dataset is co-located
//!    (`tests/nonmem/warfarin_iov.csv`, a copy of `data/warfarin_iov.csv`) so the
//!    job is self-contained: `nmfe75 warfarin_iov.ctl warfarin_iov.lst`.
//! 2. From `sdtab_iov`, take the per-`(ID, OCC)` CL (CL is constant within an
//!    occasion) and save it as `tests/nonmem/warfarin_iov_cl_reference.csv` with
//!    a `ID,OCC,CL` header (one row per subject × occasion, 20 rows here).
//! 3. Re-run with `cargo test --test warfarin_iov_nonmem -- --ignored` to
//!    activate the cross-engine comparison; tighten `CL_REL_TOL` to match the
//!    observed agreement.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{
    fit, read_nonmem_csv, EstimationMethod, FitOptions, FitResult, GradientMethod, Population,
};
use std::collections::BTreeMap;
use std::path::Path;

// NONMEM 7.5.1 FOCEI MLE (run.ext final iteration; OFV without constant).
const NM_TVCL: f64 = 0.172776;
const NM_TVV: f64 = 8.62821;
const NM_TVKA: f64 = 1.17856;
const NM_OMEGA_CL: f64 = 0.0399349;
const NM_OMEGA_V: f64 = 0.0107782;
const NM_OMEGA_KA: f64 = 0.0254197;
const NM_OMEGA_IOV: f64 = 0.0357084;
const NM_SIGMA_PROP_SD: f64 = 0.188116; // sqrt(0.0353877)
const NM_OFV_NO_CONST: f64 = 308.8305;

#[test]
#[ignore = "NONMEM-anchored IOV cross-check (issues #101/#104/#109): asserts ferx's FOCEI IOV marginal matches NONMEM to ~0.6 OFV units; needs the fixed-MLE fixture"]
fn iov_objective_matches_nonmem() {
    // examples/warfarin_iov.ferx structural model, parameters FIXed at NONMEM's
    // MLE. omega/kappa are variances; sigma is the SD ferx reports.
    let fixed = format!(
        r"
[parameters]
  theta TVCL({NM_TVCL}, FIX)
  theta TVV({NM_TVV}, FIX)
  theta TVKA({NM_TVKA}, FIX)
  omega ETA_CL ~ {NM_OMEGA_CL} FIX
  omega ETA_V  ~ {NM_OMEGA_V} FIX
  omega ETA_KA ~ {NM_OMEGA_KA} FIX
  kappa KAPPA_CL ~ {NM_OMEGA_IOV} FIX
  sigma PROP_ERR ~ {prop} (sd) FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  iov_column = OCC
",
        prop = NM_SIGMA_PROP_SD,
    );

    let model = parse_model_string(&fixed).expect("fixed-param IOV model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data loads");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true; // match NONMEM METHOD=1 INTER
    opts.gradient_method = GradientMethod::Fd;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts)
        .expect("fixed-param IOV objective evaluation must run");

    // After the Almquist 2015 Laplace marginal switch (commit 2de0bea, closing
    // issue #109), ferx's FOCEI IOV objective matches NONMEM to ~0.6 units. The
    // prediction is exact (ferx PRED == NONMEM PRED to 5 s.f.); the remaining gap
    // is within NONMEM's own non-clean convergence on this dataset. If this band
    // breaks, the IOV marginal moved away from NONMEM — a regression to investigate.
    let diff = (NM_OFV_NO_CONST - result.ofv).abs(); // expected ≈ 0.6
    assert!(
        result.ofv.is_finite() && diff < 3.0,
        "ferx FOCEI at NONMEM's MLE = {:.4}; NONMEM = {:.4}; |gap| {:.4} exceeds the \
         expected agreement tolerance (3.0 units)",
        result.ofv,
        NM_OFV_NO_CONST,
        diff
    );
}

/// Relative tolerance for the ferx-vs-NONMEM per-`(ID, OCC)` CL comparison.
/// ferx and NONMEM compute the EBEs (η̂, κ̂) independently and their IOV
/// marginals differ slightly (Sheiner–Beal vs Laplace; ~0.6 OFV units — see
/// `iov_objective_matches_nonmem`), so individual CL can drift a little.
/// NONMEM 7.5.1 FOCEI (warfarin IOV, 10 subjects × 2 occasions):
/// all 20 (ID, OCC) pairs within 6.6% (worst ID=3 OCC=1), band set to 10%.
const CL_REL_TOL: f64 = 0.10;

/// Per-`(ID, occasion)` individual CL from a fit's `[output] CL` column. CL is
/// constant within an occasion (one shared kappa), so the first observation of
/// each occasion is representative. Keyed `(id, occ)` for an order-independent,
/// engine-agnostic comparison.
fn ferx_cl_by_id_occ(pop: &Population, result: &FitResult) -> BTreeMap<(String, u32), f64> {
    let mut out = BTreeMap::new();
    for (subj, sr) in pop.subjects.iter().zip(result.subjects.iter()) {
        let cl = sr
            .extra_columns
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case("CL"))
            .map(|(_, v)| v)
            .expect("[output] CL column present in SubjectResult");
        for (j, &occ) in subj.occasions.iter().enumerate() {
            out.entry((subj.id.clone(), occ)).or_insert(cl[j]);
        }
    }
    out
}

/// Parse an `ID,OCC,CL` reference table (header row optional, columns trimmed).
/// Returns the per-`(ID, OCC)` CL NONMEM emitted in `sdtab_iov`.
fn parse_nonmem_cl_reference(path: &Path) -> BTreeMap<(String, u32), f64> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read NONMEM CL reference {}: {e}", path.display()));
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let cols: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if cols.len() < 3 {
            continue;
        }
        // A header row (non-numeric OCC/CL) parses to None and is skipped.
        if let (Ok(occ), Ok(cl)) = (cols[1].parse::<f64>(), cols[2].parse::<f64>()) {
            out.insert((cols[0].to_string(), occ.round() as u32), cl);
        }
    }
    assert!(
        !out.is_empty(),
        "NONMEM CL reference {} parsed to zero rows — expected 'ID,OCC,CL'",
        path.display()
    );
    out
}

// NOT `#[ignore]`d: this runs in PR CI via the `test` job's dedicated NONMEM
// cross-check step (`cargo test --test warfarin_iov_nonmem`), since the per-PR
// `cargo test --lib` does not build integration tests. The reference fixture
// (tests/nonmem/warfarin_iov_cl_reference.csv) is committed and the FIXed-MLE
// fit is sub-second. Part B no-ops with a clear message only if that fixture is
// somehow absent; with it present (the committed state) the cross-engine
// comparison runs and gates the PR.
#[test]
fn iov_individual_cl_matches_nonmem() {
    // examples/warfarin_iov.ferx structural model, parameters FIXed at NONMEM's
    // MLE, with CL/V/KA echoed as [output] columns so the post-fit per-occasion
    // individual parameters are materialised in SubjectResult::extra_columns
    // (the path fixed in issue #238).
    let fixed = format!(
        r"
[parameters]
  theta TVCL({NM_TVCL}, FIX)
  theta TVV({NM_TVV}, FIX)
  theta TVKA({NM_TVKA}, FIX)
  omega ETA_CL ~ {NM_OMEGA_CL} FIX
  omega ETA_V  ~ {NM_OMEGA_V} FIX
  omega ETA_KA ~ {NM_OMEGA_KA} FIX
  kappa KAPPA_CL ~ {NM_OMEGA_IOV} FIX
  sigma PROP_ERR ~ {prop} (sd) FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[output]
  CL
  V
  KA

[fit_options]
  method     = foce
  iov_column = OCC
",
        prop = NM_SIGMA_PROP_SD,
    );

    let model = parse_model_string(&fixed).expect("fixed-param IOV model with [output] parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data loads");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true; // match NONMEM METHOD=1 INTER
    opts.gradient_method = GradientMethod::Fd;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts)
        .expect("fixed-param IOV fit with [output] columns must run");

    // ── Part A: internal consistency (no NONMEM reference required) ──
    // Issue #238: the post-fit CL column must carry each observation's *occasion*
    // kappa. Equivalent observable facts at FIXed params:
    //   • V and KA (BSV only) are identical across all of a subject's obs.
    //   • CL is constant *within* an occasion and differs *between* occasions
    //     wherever the occasion kappas differ. The pre-#238 bug (kappa silently 0)
    //     makes CL occasion-independent for *every* subject.
    let col = |sr: &ferx_core::SubjectResult, name: &str| -> Vec<f64> {
        sr.extra_columns
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| panic!("[output] {name} column present"))
    };
    let mut any_cl_varies_by_occ = false;
    for (subj, sr) in pop.subjects.iter().zip(result.subjects.iter()) {
        let cl = col(sr, "CL");
        let v = col(sr, "V");
        let ka = col(sr, "KA");
        for j in 1..v.len() {
            assert!(
                (v[j] - v[0]).abs() < 1e-9 * v[0].abs().max(1.0),
                "V varies within subject {} — V has no IOV",
                subj.id
            );
            assert!(
                (ka[j] - ka[0]).abs() < 1e-9 * ka[0].abs().max(1.0),
                "KA varies within subject {} — KA has no IOV",
                subj.id
            );
        }
        // CL constant within each occasion; collect the distinct per-occasion CL.
        let mut occ_cl: BTreeMap<u32, f64> = BTreeMap::new();
        for (j, &occ) in subj.occasions.iter().enumerate() {
            match occ_cl.get(&occ) {
                Some(&c0) => assert!(
                    (cl[j] - c0).abs() < 1e-9 * c0.abs().max(1.0),
                    "CL varies within occasion {occ} of subject {} — should share one kappa",
                    subj.id
                ),
                None => {
                    occ_cl.insert(occ, cl[j]);
                }
            }
        }
        let cls: Vec<f64> = occ_cl.values().copied().collect();
        if cls.len() >= 2
            && cls
                .windows(2)
                .any(|w| (w[0] - w[1]).abs() > 1e-6 * w[0].abs().max(1.0))
        {
            any_cl_varies_by_occ = true;
        }
    }
    assert!(
        any_cl_varies_by_occ,
        "no subject's CL differed across occasions — per-occasion kappa is not reaching the \
         post-fit CL column (regression of issue #238)"
    );

    // ── Part B: cross-engine check against NONMEM ──
    // The reference fixture is committed and this test gates every PR (see the
    // dedicated CI step), so a missing fixture is a hard failure — never a silent
    // skip that would let the cross-engine comparison quietly stop running.
    let ref_path = Path::new("tests/nonmem/warfarin_iov_cl_reference.csv");
    assert!(
        ref_path.exists(),
        "NONMEM CL reference missing at {} — it is a committed fixture this test depends on. \
         Regenerate it by running tests/nonmem/warfarin_iov.ctl (NONMEM 7.5.1 FOCEI) and saving \
         the per-(ID,OCC) CL from sdtab_iov as 'ID,OCC,CL' (see module docs).",
        ref_path.display()
    );
    let nm = parse_nonmem_cl_reference(ref_path);
    let fx = ferx_cl_by_id_occ(&pop, &result);
    let mut worst = 0.0_f64;
    for ((id, occ), &cl_nm) in &nm {
        let cl_fx = *fx
            .get(&(id.clone(), *occ))
            .unwrap_or_else(|| panic!("ferx produced no CL for ID={id} OCC={occ}"));
        let rel = (cl_fx - cl_nm).abs() / cl_nm.abs().max(1e-12);
        worst = worst.max(rel);
        assert!(
            rel < CL_REL_TOL,
            "ID={id} OCC={occ}: ferx CL={cl_fx:.6}, NONMEM CL={cl_nm:.6}, rel diff {rel:.4} \
             exceeds CL_REL_TOL ({CL_REL_TOL})"
        );
    }
    eprintln!(
        "ferx vs NONMEM per-(ID,OCC) CL: {} pairs agree within {:.1}% (worst {:.3}%)",
        nm.len(),
        CL_REL_TOL * 100.0,
        worst * 100.0
    );
}
