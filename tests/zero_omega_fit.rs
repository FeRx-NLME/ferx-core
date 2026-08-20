//! Tier-2 integration tests for fixed-effects-only models (`n_eta = 0`, #989).
//!
//! These exercise the public `fit()` boundary and return immediately —
//! `outer_maxiter = 3`, no convergence loop — so they run in the PR job without a
//! `slow-tests` gate. The full-convergence NONMEM anchor lives in
//! `tests/zero_omega_nonmem_anchor.rs` (Tier 3).
//!
//! The Gaussian and the TTE zero-Omega paths are both covered here on purpose.
//! Before #989 the empty-Omega branch was gated on the *endpoint type*, so a
//! survival model reached it and a PK model did not; now they share one code
//! path, and a regression in either must not hide behind the other.

use ferx_core::{fit_from_files, FitOptions};

fn smoke_opts() -> FitOptions {
    FitOptions {
        outer_maxiter: 3,
        run_covariance_step: false,
        ..Default::default()
    }
}

/// A continuous (residual-error) model with no `omega` declarations fits.
///
/// This is the shape from the #989 report: before the fix, parsing failed with
/// `No omega parameters defined` and `fit()` was never reached.
#[test]
fn continuous_model_with_no_random_effects_fits() {
    let res = fit_from_files(
        "examples/one_cpt_iv_pooled.ferx",
        Some("data/one_cpt_iv.csv"),
        None,
        Some(smoke_opts()),
    )
    .expect("a fixed-effects-only continuous fit must return Ok");

    assert!(res.ofv.is_finite(), "OFV must be finite, got {}", res.ofv);
    assert_eq!(res.omega.nrows(), 0, "Omega must be 0x0");
    assert_eq!(res.omega.ncols(), 0, "Omega must be 0x0");
    assert!(res.eta_names.is_empty());
    // Every eta-indexed output must be empty rather than a zero-length-but-present
    // artefact or a NaN: nothing downstream should invent a random effect.
    assert!(
        res.shrinkage_eta.is_empty(),
        "eta shrinkage must be empty, got {:?}",
        res.shrinkage_eta
    );
    assert!(res.omega_iov.is_none(), "no IOV was declared");
    // Sigma is still doing all the work — it is what the likelihood is built on.
    assert_eq!(res.sigma.len(), 1);
    assert!(res.sigma[0].is_finite() && res.sigma[0] > 0.0);
}

/// With no random effects the individual prediction *is* the population
/// prediction, so the two residual flavours must coincide exactly.
///
/// This is the cheapest end-to-end check that the FOCE/FOCEI machinery really did
/// collapse rather than quietly carrying a phantom eta: any non-empty conditional
/// step would move `IPRED` off `PRED`.
#[test]
fn no_random_effects_makes_ipred_equal_pred_and_cwres_equal_iwres() {
    let res = fit_from_files(
        "examples/one_cpt_iv_pooled.ferx",
        Some("data/one_cpt_iv.csv"),
        None,
        Some(smoke_opts()),
    )
    .expect("fit must return Ok");

    let mut checked = 0usize;
    for subj in &res.subjects {
        assert!(
            subj.eta.is_empty(),
            "a subject must carry no EBEs when n_eta = 0, got {:?}",
            subj.eta
        );
        for (i, (&pred, &ipred)) in subj.pred.iter().zip(subj.ipred.iter()).enumerate() {
            assert_eq!(
                pred.to_bits(),
                ipred.to_bits(),
                "PRED != IPRED at obs {i}: {pred} vs {ipred}"
            );
            checked += 1;
        }
        for (i, (&cwres, &iwres)) in subj.cwres.iter().zip(subj.iwres.iter()).enumerate() {
            assert_eq!(
                cwres.to_bits(),
                iwres.to_bits(),
                "CWRES != IWRES at obs {i}: {cwres} vs {iwres}"
            );
        }
    }
    assert!(checked > 0, "fixture produced no observations to compare");
}

/// SAEM has no latent variable to integrate at `n_eta = 0`, so it is rejected —
/// and rejected up front, by the same `check_model_options` gauntlet `ferx check`
/// runs, rather than after a stage of the chain has already executed.
#[test]
fn saem_is_rejected_without_random_effects() {
    let opts = FitOptions {
        methods: vec![ferx_core::types::EstimationMethod::Saem],
        ..smoke_opts()
    };
    let err = fit_from_files(
        "examples/one_cpt_iv_pooled.ferx",
        Some("data/one_cpt_iv.csv"),
        None,
        Some(opts),
    )
    .expect_err("saem at n_eta = 0 must be rejected");
    assert!(
        err.contains("at least one random effect"),
        "unexpected message: {err}"
    );
}

/// The survival half of the pair: a TTE endpoint with neither Omega nor sigma.
///
/// This path already shipped before #989, which is exactly why it needs a pin —
/// the fix removed the endpoint-type condition that used to be the *only* thing
/// letting it through.
#[cfg(feature = "survival")]
#[test]
fn tte_model_with_no_random_effects_and_no_sigma_fits() {
    use ferx_core::parser::model_parser::parse_model_string;
    use std::io::Write;

    const POOLED_TTE: &str = r"
[parameters]
  theta TVSCALE(20.0, 0.1, 500.0)
  theta TVSHAPE(1.5,  0.1, 10.0)

[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE
  shape  = TVSHAPE
";
    // Ω and σ are separate capabilities and this model drops both: a pure-TTE
    // endpoint has no residual-error term at all, whereas the continuous tests
    // above keep sigma and drop only Ω.
    let model =
        parse_model_string(POOLED_TTE).expect("a TTE model with no omega and no sigma must parse");
    assert_eq!(model.n_eta, 0);
    assert!(
        model.default_params.sigma.values.is_empty(),
        "a pure-TTE model carries no sigma"
    );

    // Go through `fit_from_files` rather than `read_nonmem_csv` + `fit`: the TTE
    // rows on CMT 2 only route to the event likelihood when the population is read
    // through the endpoint-aware chokepoint, which is what the CLI and the R
    // wrapper both use. Reading the CSV blind makes CMT 2 look like an
    // observation compartment with no `[error_model]` entry.
    let mut f = tempfile::Builder::new()
        .suffix(".ferx")
        .tempfile()
        .expect("create temp model");
    f.write_all(POOLED_TTE.as_bytes())
        .expect("write temp model");
    f.flush().expect("flush temp model");

    let res = fit_from_files(
        f.path().to_str().expect("temp path is utf-8"),
        Some("data/tte_weibull.csv"),
        None,
        Some(smoke_opts()),
    )
    .expect("a fixed-effects-only TTE fit must return Ok");
    assert!(res.ofv.is_finite(), "OFV must be finite, got {}", res.ofv);
    assert_eq!(res.omega.nrows(), 0);
    assert!(res.shrinkage_eta.is_empty());
    assert!(res.sigma.is_empty(), "a pure-TTE fit reports no sigma");
}

// --- IOV without BSV --------------------------------------------------------
//
// `n_eta = 0` with `n_kappa > 0` is a model class #989 unblocked on its own: the
// removed rejection keyed on the **BSV** eta list, so a kappa-only model was
// refused even though its IOV Omega was well-formed. `parse_full_model` coverage
// lives in `src/parser/model_parser_tests.rs`; what needs an *oracle* is the
// marginal, because the FOCE augmented prior at `n_eta = 0` is
// `blkdiag(Ω_bsv (0x0), Ω_iov x K)` and a silently-dropped kappa penalty would
// still produce a finite, plausible OFV.
//
// There is no NONMEM run to anchor a kappa-only fit against as a single object,
// so this uses the **degenerate oracle** CLAUDE.md prescribes for that case.
// Collapse the data to one occasion per subject and a kappa is, term for term, a
// BSV eta: one draw per subject from one prior. The two objectives must therefore
// agree exactly at a shared point, and they do — bit-for-bit, not to a tolerance.

/// `data/warfarin_iov.csv` with every row's `OCC` forced to 1, so each subject
/// has exactly one occasion.
fn single_occasion_warfarin() -> tempfile::NamedTempFile {
    use std::io::Write;
    let src = std::fs::read_to_string("data/warfarin_iov.csv").expect("read warfarin_iov.csv");
    let mut lines = src.lines();
    let header = lines.next().expect("csv has a header");
    let occ = header
        .split(',')
        .position(|c| c.trim() == "OCC")
        .expect("warfarin_iov.csv carries an OCC column");

    let mut out = String::with_capacity(src.len());
    out.push_str(header);
    out.push('\n');
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let mut cells: Vec<&str> = line.split(',').collect();
        cells[occ] = "1";
        out.push_str(&cells.join(","));
        out.push('\n');
    }

    let mut f = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("create temp data");
    f.write_all(out.as_bytes()).expect("write temp data");
    f.flush().expect("flush temp data");
    f
}

/// The two models differ only in whether the single random effect is declared as
/// BSV or as IOV; `{RE}` is substituted with the declaration and `{TERM}` with the
/// name used in `CL`. `maxiter = 0` makes this a pure evaluation at the shared
/// starting values, and `checkpoint = false` keeps it from writing a `.tmp`.
fn one_effect_model(decl: &str, term: &str, extra_opts: &str) -> tempfile::NamedTempFile {
    use std::io::Write;
    let src = format!(
        "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  {decl}
  sigma PROP_ERR ~ 0.2 (sd)

[individual_parameters]
  CL = TVCL * exp({term})
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  maxiter    = 0
  covariance = false
  checkpoint = false
{extra_opts}"
    );
    let mut f = tempfile::Builder::new()
        .suffix(".ferx")
        .tempfile()
        .expect("create temp model");
    f.write_all(src.as_bytes()).expect("write temp model");
    f.flush().expect("flush temp model");
    f
}

#[test]
fn iov_without_bsv_reduces_to_bsv_on_a_single_occasion() {
    use ferx_core::run_model_with_data;

    let data = single_occasion_warfarin();
    let data_path = data.path().to_str().expect("temp data path is utf-8");

    let bsv = one_effect_model("omega ETA_CL ~ 0.09", "ETA_CL", "");
    let iov = one_effect_model("kappa KAPPA_CL ~ 0.09", "KAPPA_CL", "  iov_column = OCC\n");

    let (bsv_res, _) = run_model_with_data(
        bsv.path().to_str().expect("temp model path is utf-8"),
        Some(data_path),
    )
    .expect("the BSV reference fit must return Ok");
    let (iov_res, _) = run_model_with_data(
        iov.path().to_str().expect("temp model path is utf-8"),
        Some(data_path),
    )
    .expect("a kappa-only fit must return Ok (#989)");

    assert_eq!(
        bsv_res.omega.nrows(),
        1,
        "the reference must carry exactly one BSV eta"
    );
    assert_eq!(
        iov_res.omega.nrows(),
        0,
        "the IOV model must carry no BSV Omega"
    );
    assert_eq!(
        iov_res
            .omega_iov
            .as_ref()
            .expect("a declared kappa must yield an IOV Omega")
            .nrows(),
        1
    );

    // Not bit-for-bit, and it cannot be: the IOV marginal builds its kappa columns
    // of H by central differences (`foce_subject_nll_iov`, EPS = 1e-6) where the BSV
    // path gets its eta columns from the shared Jacobian, so the two log|H̃| terms
    // differ in the last few ULP of the stencil. Observed delta 1.3e-7 on ~553.
    //
    // The band is what makes this an oracle rather than a smoke test: dropping the
    // kappa prior — the failure this guards against — costs `½·log|Ω_iov|` per
    // occasion plus the Mahalanobis term, ~24 OFV units for these 10 subjects at
    // Ω_iov = 0.09. Anything that silently loses the penalty misses by seven orders
    // of magnitude more than the stencil noise.
    let delta = (bsv_res.ofv - iov_res.ofv).abs();
    assert!(
        delta < 1e-6,
        "one occasion makes a kappa a BSV eta, so the objectives must agree: \
         BSV {} vs IOV-only {} (delta {:.3e})",
        bsv_res.ofv,
        iov_res.ofv,
        delta
    );
}
