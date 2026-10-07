//! #1755: the SIR draw scorer is the fit's own objective.
use super::*;
use crate::estimation::fit_inputs::fitted_marginal_options;
use crate::estimation::parameterization::unpack_params;
use crate::estimation::uncertainty_samples::fitted_params_from_result;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// ferx's Laplace optimum on `examples/warfarin.ferx`, packed — the same point as
/// `tests/sir_scores_fitted_marginal.rs`'s `WARF_LAPLACE_PACKED`.
const WARF_LAPLACE_PACKED: [f64; 7] = [
    -2.019760138501293,
    2.0460733046385786,
    -0.20960659528716724,
    -1.7773234722669555,
    -2.3234007261778418,
    -0.5452958750862014,
    -4.550252146959798,
];

/// `tests/mixture_nonmem.rs`'s model with Ω/Σ free and a class-2 Ω override, and its
/// FOCEI optimum on `tests/nonmem/mixture_iv.csv` (OFV 298.3278664427295).
const MIX_OVERRIDE: &str = r"
[parameters]
  theta TVCL1(1.2, 0.01, 100.0)
  theta TVCL2(2.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.04

[mixture]
  nsub = 2
  logit(1) = MIXL
  omega(2) ETA_CL ~ 0.30

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
const MIX_PACKED: [f64; 7] = [
    -0.07780620268043612,
    0.993379148024101,
    2.3028646917499356,
    -0.30316582094104605,
    -1.9090553430428145,
    -1.5518494729012504,
    -1.1652410428760827,
];

fn mixture_inputs() -> (CompiledModel, Population) {
    let model = crate::parse_model_string(MIX_OVERRIDE).expect("mixture model must parse");
    let pop = crate::read_nonmem_csv(
        &root().join("tests/nonmem/mixture_iv.csv"),
        Some(&["WT"]),
        None,
    )
    .expect("mixture data must load");
    (model, pop)
}

/// The scorer at the fit's estimates, with `run_sir`'s scoring options built from
/// caller defaults, minus the fit's data OFV. Bitwise 0 is the claim.
fn scorer_gap(model: &CompiledModel, pop: &Population, fit: &FitResult) -> (f64, f64) {
    let options = fitted_marginal_options(fit, &FitOptions::default());
    let params = fitted_params_from_result(fit, model).expect("params rebuild");
    let etas: Vec<DVector<f64>> = fit.subjects.iter().map(|s| s.eta.clone()).collect();
    let got = sir_draw_ofv(model, pop, &params, &etas, &options);
    (got, fit.ofv - fit.ofv_prior)
}

/// T3. `sir_draw_ofv` at the estimates **is** the fit's objective — bit for bit on
/// FOCEI, Laplace and a `[mixture]` override fit, to 1e-8 on FOCE — — scored through `run_sir`'s option
/// builder with caller defaults (`method = FoceI`). The FOCE / FOCEI arms are the
/// no-move control; the Laplace arm dies when the builder stops taking the fit's
/// method (−3.40e-2 on the #1755 probe), the mixture arm when the scorer loses its
/// mixture branch (+639.6).
#[test]
fn sir_draw_ofv_at_the_estimate_is_the_fits_objective() {
    let prep = crate::api::prepare_run(
        root().join("examples/warfarin.ferx").to_str().unwrap(),
        Some(root().join("data/warfarin.csv").to_str().unwrap()),
    )
    .expect("warfarin must prepare");
    let init = unpack_params(&WARF_LAPLACE_PACKED, &prep.init_params);
    let mut gaps: Vec<(String, f64, f64)> = Vec::new();
    for method in [
        EstimationMethod::Foce,
        EstimationMethod::FoceI,
        EstimationMethod::Laplace,
    ] {
        let opts = FitOptions {
            method,
            outer_maxiter: 0,
            run_covariance_step: false,
            verbose: false,
            ..FitOptions::default()
        };
        let fit = crate::api::fit(&prep.parsed.model, &prep.population, &init, &opts)
            .expect("warfarin fit must run");
        let (got, want) = scorer_gap(&prep.parsed.model, &prep.population, &fit);
        gaps.push((format!("{method:?}"), got, want));
    }

    let (model, pop) = mixture_inputs();
    let init = unpack_params(&MIX_PACKED, &model.default_params);
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let fit = crate::api::fit(&model, &pop, &init, &opts).expect("mixture fit must run");
    let (got, want) = scorer_gap(&model, &pop, &fit);
    gaps.push(("mixture".into(), got, want));

    for (arm, got, want) in &gaps {
        eprintln!(
            "MEASURE T3 {arm} scorer={got:?} fit={want:?} gap={:e}",
            got - want
        );
    }
    for (arm, got, want) in &gaps {
        assert!(got.is_finite() && want.is_finite(), "{arm}: non-finite");
        if arm == "Foce" {
            // The one arm that is not bitwise: FOCE's frozen-variance EBEs (#1722),
            // re-solved warm from the reported η̂, land 4.24e-10 away (Linux aarch64 and
            // macOS arm64 alike) —
            // the FOCE scorer is not what this PR changes, and it is the no-move
            // control. 1e-8 is 24× that and 3.4e6× below the Laplace mutation.
            assert!(
                (got - want).abs() <= 1e-8,
                "{arm}: SIR scorer {got} vs the fit's objective {want} (gap {:e})",
                got - want
            );
        } else {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{arm}: SIR scorer {got} vs the fit's objective {want} (gap {:e})",
                got - want
            );
        }
    }
}

/// T8. `run_sir_core` refuses parameters whose mixture shape disagrees with the model —
/// both directions — with an `Err`, not the `mixture_ofv` panic the scorer would hit.
#[test]
fn run_sir_core_refuses_mixture_params_mismatch() {
    let (model, pop) = mixture_inputs();
    let cov = DMatrix::identity(7, 7);
    let etas = vec![DVector::zeros(1); pop.subjects.len()];
    let opts = FitOptions {
        verbose: false,
        ..FitOptions::default()
    };

    let mut flat = model.default_params.clone();
    flat.mixture = None;
    let e = run_sir_core(&model, &pop, &flat, &etas, &cov, 0.0, &opts)
        .expect_err("a [mixture] model with one-class parameters must be refused");
    assert!(
        e.contains("the model declares [mixture] but the parameters carry no per-class"),
        "{e}"
    );

    // The same model without its `[mixture]` block.
    let plain = crate::parse_model_string(
        &MIX_OVERRIDE
            .replace(
                "[mixture]\n  nsub = 2\n  logit(1) = MIXL\n  omega(2) ETA_CL ~ 0.30\n",
                "",
            )
            .replace(
                "if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)",
                "TVCL1 * exp(ETA_CL)",
            ),
    )
    .expect("non-mixture variant must parse");
    assert!(plain.mixture.is_none(), "fixture premise: no [mixture]");
    let e = run_sir_core(&plain, &pop, &model.default_params, &etas, &cov, 0.0, &opts)
        .expect_err("per-class parameters on a non-mixture model must be refused");
    assert!(e.contains("the model declares no [mixture]"), "{e}");
}
