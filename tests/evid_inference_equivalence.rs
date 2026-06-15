//! #262 end-to-end: a dataset with **no `EVID` column** must fit *identically*
//! to the same dataset with an explicit `EVID` column, because ferx infers a
//! dose from a nonzero `AMT` exactly where NONMEM does.
//!
//! `data/warfarin.csv` is "inference-clean" — every `EVID=1` row carries
//! `AMT>0`, every `EVID=0` row carries `AMT=0`, and there are no `EVID` 2/3/4
//! rows — so removing the `EVID` column reproduces the identical event stream
//! (10 doses, 110 observations) and therefore the identical fit. This is the
//! NONMEM anchor for the dose-inference path: warfarin's *with-EVID* FOCEI fit
//! is the NONMEM-validated reference (see `warfarin_covariance_nonmem.rs`,
//! OFV ≈ −280 vs NONMEM −280.36), so matching it proves the inferred-dose fit
//! lands on the NONMEM result without any manual `EVID` editing.
//!
//! Tier-3: runs two fits to convergence, so it is `#[ignore]`d outside the
//! `slow-tests` feature (nightly / on-demand).

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::io::Write;
use std::path::Path;

// Warfarin 1-cpt oral, proportional error — the same structural model the
// NONMEM covariance cross-check uses.
const MODEL_SRC: &str = r"
[parameters]
  theta TVCL(0.15, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.2, 0.01, 50.0)
  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Write a copy of `data/warfarin.csv` with the `EVID` column removed. Returns
/// the temp file; the caller must keep it alive for the path to stay valid.
fn warfarin_without_evid() -> tempfile::NamedTempFile {
    let src = std::fs::read_to_string("data/warfarin.csv").expect("read data/warfarin.csv");
    let header = src.lines().next().expect("warfarin.csv has a header");
    let evid_idx = header
        .split(',')
        .position(|h| h.trim().eq_ignore_ascii_case("evid"))
        .expect("warfarin.csv has an EVID column");

    let drop_evid = |row: &str| -> String {
        row.split(',')
            .enumerate()
            .filter(|(i, _)| *i != evid_idx)
            .map(|(_, c)| c)
            .collect::<Vec<_>>()
            .join(",")
    };

    let mut out = String::new();
    for line in src.lines() {
        out.push_str(&drop_evid(line));
        out.push('\n');
    }

    let mut f = tempfile::NamedTempFile::new().expect("create temp csv");
    f.write_all(out.as_bytes()).expect("write temp csv");
    f
}

/// Fit warfarin from `data_path`, returning (OFV, total dose events parsed).
fn fit_warfarin(data_path: &Path) -> (f64, usize) {
    let model = parse_model_string(MODEL_SRC).expect("warfarin model parses");
    let pop = read_nonmem_csv(data_path, None, None).expect("warfarin data loads");
    let total_doses: usize = pop.subjects.iter().map(|s| s.doses.len()).sum();

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true;
    opts.outer_maxiter = 300;
    opts.verbose = false;

    let result = fit(&model, &pop, &model.default_params, &opts).expect("warfarin fit runs");
    (result.ofv, total_doses)
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn no_evid_warfarin_fits_identically_to_explicit_evid() {
    // Explicit EVID — the NONMEM-validated reference fit.
    let (ofv_with, doses_with) = fit_warfarin(Path::new("data/warfarin.csv"));
    // EVID column stripped — doses inferred from AMT.
    let no_evid = warfarin_without_evid();
    let (ofv_without, doses_without) = fit_warfarin(no_evid.path());

    // The reference parses warfarin's 10 doses (guards against a silently empty
    // stripped file) ...
    assert_eq!(
        doses_with, 10,
        "explicit-EVID warfarin should parse 10 doses"
    );
    // ... and inference reproduces exactly those doses (not zero — the #262 bug).
    assert_eq!(
        doses_without, doses_with,
        "stripping the EVID column must infer the same dose events"
    );

    // Same event stream → same fit. The tolerance only absorbs floating-point
    // reduction noise; a regression that dropped the inferred doses would land on
    // the degenerate dose-free optimum (OFV in the thousands), not within 1e-4.
    assert!(
        (ofv_with - ofv_without).abs() < 1e-4,
        "no-EVID warfarin OFV {ofv_without} must match explicit-EVID OFV {ofv_with}"
    );
}
