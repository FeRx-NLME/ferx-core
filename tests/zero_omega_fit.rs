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
