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
    let (options, params, etas) = scorer_inputs(model, fit);
    let class_etas = sir_class_etas(model, pop, &params, &options);
    let got = sir_draw_ofv(
        model,
        pop,
        &params,
        &params,
        &etas,
        class_etas.as_deref(),
        &options,
    )
    .expect("the scorer at the estimates rejects no inner start");
    (got, fit.ofv - fit.ofv_prior)
}

fn scorer_inputs(
    model: &CompiledModel,
    fit: &FitResult,
) -> (FitOptions, ModelParameters, Vec<DVector<f64>>) {
    let options = fitted_marginal_options(fit, &FitOptions::default());
    let params = fitted_params_from_result(fit, model).expect("params rebuild");
    let etas: Vec<DVector<f64>> = fit.subjects.iter().map(|s| s.eta.clone()).collect();
    (options, params, etas)
}

/// The subjects whose η̂ the scorer's warm re-solve moves off the fit's, as
/// `(index, max |Δη|)`. Empty is the claim: the fit's η̂ is a fixed point (#2279).
fn moved_etas(model: &CompiledModel, pop: &Population, fit: &FitResult) -> Vec<(usize, f64)> {
    let (options, params, etas) = scorer_inputs(model, fit);
    let (ehs, _, _, _) = sir_draw_ebes(model, pop, &params, &params, &etas, &options);
    ehs.iter()
        .zip(&etas)
        .enumerate()
        .filter(|(_, (a, b))| {
            a.iter()
                .zip(b.iter())
                .any(|(x, y)| x.to_bits() != y.to_bits())
        })
        .map(|(i, (a, b))| (i, (a - b).amax()))
        .collect()
}

/// T3. `sir_draw_ofv` at the estimates **is** the fit's objective — bit for bit on
/// FOCE, FOCEI, Laplace and a `[mixture]` override fit — scored through `run_sir`'s option
/// builder with caller defaults (`method = FoceI`). The FOCE / FOCEI arms are the
/// no-move control; the Laplace arm dies when the builder stops taking the fit's
/// method (−3.40e-2 on the #1755 probe), the mixture arm when the scorer loses its
/// mixture branch (+639.6).
///
/// #2279: bitwise is a property of the code, not of rounding — the fit's η̂ is a fixed
/// point of the scorer's warm re-solve ([`sir_draw_ebes`]), asserted directly on every
/// warfarin arm. Before #2279 it was not, twice over. The re-solve was unseeded, so BFGS
/// failed its line search at η̂ and the Nelder–Mead fallback moved FOCE subjects 0 and 7 by
/// 3.9e-14 (macOS arm64): FOCE's 5.7e-13 objective gap on every platform. And the fit's cold
/// final pass leaves some subjects fallback-certified (gradient norm above `tol`, Newton step
/// below it), which a re-search stalls on and the fallback nudges: Laplace subject 8 by
/// 1.7e-14 under Linux x86_64 / glibc 2.28, a 6.8e-13 objective gap that macOS rounded back
/// to the same bits. The seed reverting reddens the η̂ assertion on macOS; the second half is
/// pinned platform-independently by `scorer_keeps_a_start_its_certificate_accepts`.
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
        let moved = moved_etas(&prep.parsed.model, &prep.population, &fit);
        assert!(
            moved.is_empty(),
            "{method:?}: the scorer's warm re-solve moved the fit's η̂ (subject, max |Δη|): \
             {moved:?}"
        );
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
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{arm}: SIR scorer {got} vs the fit's objective {want} (gap {:e})",
            got - want
        );
    }
}

/// T8. `run_sir_core` refuses parameters whose mixture shape disagrees with the model —
/// both directions — with an `Err`, not the `mixture_ofv` panic the scorer would hit.
/// #2044: as `E_PARAM_SHAPE` (no block) on the coded core, the `String` core's text
/// being its `to_string()`. Each whole text is pinned, so deleting a sentence reddens
/// its direction; either site back on `EngineError::uncoded` reddens its `code()`.
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
    let coded = |m: &CompiledModel, p: &ModelParameters| {
        let e = run_sir_core_coded(m, &pop, p, &etas, &cov, 0.0, &opts).expect_err("refused");
        let s = run_sir_core(m, &pop, p, &etas, &cov, 0.0, &opts).expect_err("refused");
        assert_eq!(e.code(), Some("E_PARAM_SHAPE"), "{e}");
        assert_eq!(e.block(), None, "{e}");
        assert_eq!(e.to_string(), s);
        s
    };
    assert_eq!(
        coded(&model, &flat),
        "run_sir_core: the model declares [mixture] but the parameters carry \
         no per-class Ω/Σ (ModelParameters::mixture is None); build them with \
         fitted_params_from_result or pass the fit's own parameters."
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
    assert_eq!(
        coded(&plain, &model.default_params),
        "run_sir_core: the parameters carry per-class Ω/Σ \
         (ModelParameters::mixture) but the model declares no [mixture]."
    );
}

/// #1774: `fitted_params_from_result`'s three `[mixture]` override refusals carry a
/// code, shared by every entry point that rebuilds a fit's parameters (`run_sir`,
/// `run_covariance`, `simulate_with_uncertainty`). Override values the fit does not
/// store are `E_FIT_INPUT_MISSING`; a packed estimate of another layout, or one that
/// no longer reproduces the reported estimates, is `E_FIT_INPUT_MODEL`. Each text is
/// pinned whole; `run_sir` carries the same refusal with its context.
///
/// Mutations — any arm back on a bare `String`, or `fit_input_missing` /
/// `fit_input_model` swapped: that row's code; a sentence deleted: its text.
#[test]
fn mixture_override_refusals_carry_their_codes() {
    let (model, pop) = mixture_inputs();
    let init = unpack_params(&MIX_PACKED, &model.default_params);
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let fit = crate::api::fit(&model, &pop, &init, &opts).expect("mixture fit must run");
    let n = fit
        .packed_estimate
        .as_ref()
        .expect("packed-space fit")
        .len();

    // Every row here is a fit without its recorded override values (#1765), so the
    // packed-estimate fallback is what is read; the recorded-values rows are
    // `mixture_override_values_refusals_carry_their_codes`.
    let mut legacy = fit.clone();
    legacy.mixture_overrides = None;
    let mut absent = legacy.clone();
    absent.packed_estimate = None;
    let mut short = legacy.clone();
    short.packed_estimate.as_mut().unwrap().pop();
    let mut stale = legacy.clone();
    stale.theta[0] *= 1.01;
    let missing = "this [mixture] fit estimates the per-class override(s) omega(2) ETA_CL, \
                   and the fit records neither their fitted values (mixture_overrides) nor a \
                   packed estimate holding them. A fit() result records them, and so does a \
                   .fitrx bundle save_fit wrote from one; a .fitrx bundle saved before #1765 \
                   from an SAEM fit or before #1815 from any other method, or written by \
                   ferx-r before FeRx-NLME/ferx-r#655, lacks them.";
    let layout = format!(
        "the fit's packed estimate has {} coordinates but this model's parameter layout has \
         {n}, so the [mixture] override values cannot be read from it. Supply the model the \
         fit was estimated with.",
        n - 1
    );
    let changed = "the fit's packed estimate does not reproduce its reported estimates \
                   (theta / Omega / Sigma / Omega_IOV / residual correlations), so the \
                   [mixture] override values it carries cannot be trusted: the estimates or \
                   the packed estimate were changed after the fit, or this is not the model \
                   the fit was estimated with.";
    let mut bad = Vec::new();
    for (what, f, code, text) in [
        (
            "absent",
            &absent,
            "E_FIT_INPUT_MISSING",
            missing.to_string(),
        ),
        ("another layout", &short, "E_FIT_INPUT_MODEL", layout),
        ("stale", &stale, "E_FIT_INPUT_MODEL", changed.to_string()),
    ] {
        let e = fitted_params_from_result(f, &model).expect_err(what);
        let got = (e.code(), e.block(), e.context(), e.to_string());
        if got != (Some(code), None, None, text.clone()) {
            bad.push(format!("{what}: {got:?}"));
        }
        let e = crate::run_sir(f, Some(&model), Some(&pop), &FitOptions::default())
            .map(|_| ())
            .expect_err(what);
        let got = (e.code(), e.context(), e.to_string());
        if got != (Some(code), Some("run_sir"), format!("run_sir: {text}")) {
            bad.push(format!("run_sir/{what}: {got:?}"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// #2279: `keep_certified_start` returns a warm start the inner certificate accepts as it
/// stands — here the Laplace fit's η̂ moved 1e-6 per coordinate, so its gradient norm is far
/// above `tol` (BFGS would step) while its Newton step is below it (`certifies_in_place`'s
/// Newton arm). Both sides of the gate on one start: with the flag the start comes back
/// bitwise; without it the search moves it. Dies if `sir_draw_ebes` drops the flag (the
/// second assertion's twin becomes the first), or if `find_ebe_solve` stops honouring it.
///
/// And the scorer raises the flag only at the parameters the start was certified at: a
/// draw one θ step away re-solves the same start and moves it, so a draw is never scored at
/// a stale η̂ nor pays the certificate's FD Hessian (review of #2279). Dies if
/// `same_point` always holds (the draw keeps the start) or never does (the estimate moves it).
#[test]
fn scorer_keeps_a_start_its_certificate_accepts() {
    use crate::estimation::inner_optimizer::{find_ebe_cached, InnerSolvePolicy};
    let prep = crate::api::prepare_run(
        root().join("examples/warfarin.ferx").to_str().unwrap(),
        Some(root().join("data/warfarin.csv").to_str().unwrap()),
    )
    .expect("warfarin must prepare");
    let model = &prep.parsed.model;
    let opts = FitOptions {
        method: EstimationMethod::Laplace,
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let init = unpack_params(&WARF_LAPLACE_PACKED, &prep.init_params);
    let fit = crate::api::fit(model, &prep.population, &init, &opts).expect("fit must run");
    let (options, params, etas) = scorer_inputs(model, &fit);
    let start: Vec<f64> = etas[0].iter().map(|e| e + 1e-6).collect();
    let solve = |keep: bool| {
        let policy = InnerSolvePolicy {
            keep_certified_start: keep,
            ..InnerSolvePolicy::for_stage(&options, model)
        };
        let r = find_ebe_cached(
            model,
            &prep.population.subjects[0],
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(&start),
            None,
            0,
            None,
            policy,
        );
        assert!(r.converged, "keep = {keep}: not converged");
        r.eta.as_slice().to_vec()
    };
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&solve(true)), bits(&start), "kept start moved");
    let searched = solve(false);
    assert_ne!(
        bits(&searched),
        bits(&start),
        "fixture premise: a search from this start moves it (gradient norm above tol)"
    );

    // The scorer itself keeps it: a draw at the estimates scored from that start returns it.
    let mut shifted = etas.clone();
    shifted[0] = DVector::from_vec(start.clone());
    let (ehs, _, _, _) = sir_draw_ebes(
        model,
        &prep.population,
        &params,
        &params,
        &shifted,
        &options,
    );
    assert_eq!(
        bits(ehs[0].as_slice()),
        bits(&start),
        "sir_draw_ebes moved a certified start at the estimates"
    );

    // A draw away from the estimates does not keep it: the gate is off, the start is searched.
    let mut draw = params.clone();
    draw.theta[0] *= 1.0 + 1e-9;
    let (ehs, _, _, _) = sir_draw_ebes(model, &prep.population, &draw, &params, &shifted, &options);
    assert_ne!(
        bits(ehs[0].as_slice()),
        bits(&start),
        "sir_draw_ebes kept a start at a draw away from the parameters it was certified at"
    );
}

/// `same_point` is bitwise in every value the scorer reads: one ULP on any of θ, Ω, Σ, a
/// `block_sigma` ρ or Ω_IOV breaks it, and a clone keeps it. Each field gets its own case so
/// dropping one comparison names it.
#[test]
fn same_point_sees_every_scored_value() {
    use crate::types::{OmegaMatrix, ResidualCorrelation};
    let prep = crate::api::prepare_run(
        root().join("examples/warfarin.ferx").to_str().unwrap(),
        Some(root().join("data/warfarin.csv").to_str().unwrap()),
    )
    .expect("warfarin must prepare");
    let mut base = prep.init_params.clone();
    base.residual_correlations = vec![ResidualCorrelation {
        sigma_i: 0,
        sigma_j: 0,
        rho: 0.25,
    }];
    base.omega_iov = Some(OmegaMatrix::from_diagonal(&[0.04], vec!["K1".into()]));
    assert!(
        same_point(&base, &base.clone()),
        "a clone is the same point"
    );
    let up = |x: &mut f64| *x = f64::from_bits(x.to_bits() + 1);
    let cases: Vec<(&str, Box<dyn Fn(&mut ModelParameters)>)> = vec![
        ("theta", Box::new(|p| up(&mut p.theta[0]))),
        ("omega", Box::new(|p| up(&mut p.omega.matrix[(0, 0)]))),
        ("sigma", Box::new(|p| up(&mut p.sigma.values[0]))),
        ("rho", Box::new(|p| up(&mut p.residual_correlations[0].rho))),
        (
            "omega_iov",
            Box::new(|p| up(&mut p.omega_iov.as_mut().unwrap().matrix[(0, 0)])),
        ),
        ("omega_iov presence", Box::new(|p| p.omega_iov = None)),
    ];
    for (what, bump) in cases {
        let mut moved = base.clone();
        bump(&mut moved);
        assert!(
            !same_point(&base, &moved),
            "{what}: one ULP still reads as the same point"
        );
    }
}

/// #1765: a fit's recorded `mixture_overrides` is read ahead of `packed_estimate`, and
/// refused — with `E_FIT_INPUT_MODEL`, whole text pinned — when it does not hold one
/// non-negative finite value per declared override. Rows:
///
/// | `mixture_overrides`          | `packed_estimate` | outcome                         |
/// |------------------------------|-------------------|---------------------------------|
/// | the fit's own                | `None`            | `Ok`, class-2 Ω = the recorded value |
/// | the fit's own, θ edited      | the fit's (stale) | `Ok`: the values do not depend on θ |
/// | `[0.0]`                      | the fit's         | `Ok`: zero is a legal override (`~ 0 FIX`) |
/// | one value too many           | the fit's         | `Err`: count                    |
/// | `[NaN]` / `[inf]` / `[-1.0]` | the fit's         | `Err`: not a non-negative finite value |
///
/// Mutations — the field's read removed: rows 1–2 refused (no packed vector / stale);
/// either check removed: its rows `Ok` (or a panic in the Cholesky); a sentence deleted:
/// its text.
#[test]
fn mixture_override_values_refusals_carry_their_codes() {
    let (model, pop) = mixture_inputs();
    let init = unpack_params(&MIX_PACKED, &model.default_params);
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let fit = crate::api::fit(&model, &pop, &init, &opts).expect("mixture fit must run");
    let recorded = fit
        .mixture_overrides
        .clone()
        .expect("a fit() of an override model records the override values");
    assert_eq!(recorded.len(), 1);
    let chol = MIX_PACKED[6].exp();
    assert_eq!(
        recorded[0].to_bits(),
        (chol * chol).to_bits(),
        "the recorded value is the class variance the packed coordinate unpacks to"
    );

    let mut no_packed = fit.clone();
    no_packed.packed_estimate = None;
    let mut edited = fit.clone();
    edited.theta[0] *= 1.01;
    for (what, f) in [
        ("no packed estimate", &no_packed),
        ("theta edited", &edited),
    ] {
        let p = fitted_params_from_result(f, &model).unwrap_or_else(|e| panic!("{what}: {e}"));
        let mix = p.mixture.expect("mixture params");
        assert_eq!(
            mix.omega[1].matrix[(0, 0)].to_bits(),
            recorded[0].to_bits(),
            "{what}: class-2 Ω is not the recorded override"
        );
        assert_eq!(
            mix.omega[0].matrix[(0, 0)].to_bits(),
            f.omega[(0, 0)].to_bits(),
            "{what}: class 1 must track the base Ω"
        );
    }

    let mut zero = fit.clone();
    zero.mixture_overrides = Some(vec![0.0]);
    let p = fitted_params_from_result(&zero, &model).expect("a zero override is legal");
    // `OmegaMatrix` construction floors a zero variance; the rebuild lands where the
    // parser puts a `~ 0 FIX` declaration.
    let declared = crate::parse_model_string(
        &MIX_OVERRIDE.replace("omega(2) ETA_CL ~ 0.30", "omega(2) ETA_CL ~ 0 FIX"),
    )
    .expect("zero override must parse");
    assert_eq!(
        p.mixture.expect("mixture params").omega[1].matrix[(0, 0)].to_bits(),
        declared.default_params.mixture.expect("mixture").omega[1].matrix[(0, 0)].to_bits(),
        "a zero override rebuilds the class variance a zero declaration has"
    );

    let count = "the fit records 2 [mixture] override value(s) but this model declares 1 \
                 per-class override(s). Supply the model the fit was estimated with.";
    let bad_value = |v: &str| {
        format!(
            "the fit's recorded [mixture] override values include {v}, which is not a \
             non-negative finite variance or sigma, so the per-class Omega/Sigma cannot \
             be rebuilt from them."
        )
    };
    let mut bad = Vec::new();
    for (what, values, text) in [
        (
            "too many",
            vec![recorded[0], recorded[0]],
            count.to_string(),
        ),
        ("NaN", vec![f64::NAN], bad_value("NaN")),
        ("infinite", vec![f64::INFINITY], bad_value("inf")),
        ("negative", vec![-1.0], bad_value("-1")),
    ] {
        let mut f = fit.clone();
        f.mixture_overrides = Some(values);
        let e = fitted_params_from_result(&f, &model).expect_err(what);
        let got = (e.code(), e.block(), e.to_string());
        if got != (Some("E_FIT_INPUT_MODEL"), None, text.clone()) {
            bad.push(format!("{what}: {got:?}"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// #1765 review r4: a fit that records `mixture_overrides` has a stored counterpart for
/// the packed vector's override segment, so `fit_packed_estimate` reuses the vector only
/// when that segment unpacks bit-equal to the recorded values. Rows on one fit: the fit's
/// own fields are `Usable`; the recorded value one ULP up is `Stale` (`run_covariance`
/// would otherwise centre on the packed vector's override while `run_sir` uses the
/// recorded one); the field absent (an R-built or pre-#1765 fit) is `Usable` on length
/// alone, as before.
///
/// Mutations — the override comparison removed: the ULP row is `Usable`; the comparison
/// applied when the field is absent: the legacy row is `Stale`.
#[test]
fn fit_packed_estimate_checks_recorded_mixture_overrides() {
    use crate::estimation::uncertainty_samples::{fit_packed_estimate, PackedEstimate};
    let (model, pop) = mixture_inputs();
    let init = unpack_params(&MIX_PACKED, &model.default_params);
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let fit = crate::api::fit(&model, &pop, &init, &opts).expect("mixture fit must run");
    let template = &model.default_params;
    assert!(
        fit.mixture_overrides.is_some(),
        "fixture premise: the fit records its override values"
    );
    let mut edited = fit.clone();
    let x = &mut edited.mixture_overrides.as_mut().unwrap()[0];
    *x = f64::from_bits(x.to_bits() + 1);
    let mut legacy = fit.clone();
    legacy.mixture_overrides = None;
    let mut wrong = Vec::new();
    for (what, f, usable) in [
        ("the fit's own", &fit, true),
        ("override one ULP up", &edited, false),
        ("no recorded overrides", &legacy, true),
    ] {
        let got = fit_packed_estimate(f, template);
        let ok = if usable {
            matches!(got, PackedEstimate::Usable(_))
        } else {
            got == PackedEstimate::Stale
        };
        if !ok {
            wrong.push(format!("{what}: {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// #1765 review: SAEM holds every per-class Σ override at its initial value and marks it
/// `FIX` (later stages of the chain inherit the flag), so the parameters rebuilt from a fit
/// whose chain ran SAEM carry the overrides `FIX`, and the standalone covariance / SIR steps
/// treat as fixed the coordinates the in-fit ones did. Both sides of the gate on one fit:
/// with no SAEM in the chain the template's flags (free) stand; with SAEM last, or SAEM
/// earlier in the chain, every Σ override is `FIX`. The Ω override's flag never moves.
///
/// Mutations — the gate removed: the SAEM rows keep `false`; the gate always on: the
/// FOCEI row turns `true`; the gate reading only `method`: the `[saem, focei]` row.
#[test]
fn saem_chain_holds_sigma_overrides_fixed_in_rebuilt_params() {
    let src = MIX_OVERRIDE.replace(
        "  omega(2) ETA_CL ~ 0.30\n",
        "  omega(2) ETA_CL ~ 0.30\n  sigma(2) EPS ~ 0.09\n",
    );
    let model = crate::parse_model_string(&src).expect("sigma-override model must parse");
    let tmpl = model.default_params.mixture.as_ref().expect("mixture");
    assert_eq!(
        (
            tmpl.omega_override_fixed.clone(),
            tmpl.sigma_override_fixed.clone()
        ),
        (vec![false], vec![false]),
        "fixture premise: both overrides are declared free"
    );
    let (_, pop) = mixture_inputs();
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let fit = crate::api::fit(&model, &pop, &model.default_params, &opts).expect("fit runs");
    assert_eq!(fit.mixture_overrides.as_ref().map(Vec::len), Some(2));
    let flags = |method: EstimationMethod, chain: Vec<EstimationMethod>| {
        let mut f = fit.clone();
        f.method = method;
        f.method_chain = chain;
        let mix = fitted_params_from_result(&f, &model)
            .expect("rebuild")
            .mixture
            .expect("mixture");
        (mix.omega_override_fixed, mix.sigma_override_fixed)
    };
    use EstimationMethod::{FoceI, Saem};
    for (what, method, chain, want) in [
        ("focei", FoceI, vec![FoceI], false),
        ("saem", Saem, vec![Saem], true),
        ("saem then focei", FoceI, vec![Saem, FoceI], true),
    ] {
        assert_eq!(
            flags(method, chain),
            (vec![false], vec![want]),
            "{what}: override FIX flags"
        );
    }
}

/// #1769: a `[mixture]` draw away from the estimates is scored warm, every class started
/// from its own η̂ at the estimates ([`sir_class_etas`]); a draw at the estimates is scored
/// cold, the pass the fit's objective comes from (T3's bitwise claim).
///
/// The warm and cold objectives differ only at the inner tolerance (−1.3e-7 at the draw
/// below), so both sides of the `same_point` gate are asserted bitwise, each against the
/// twin it must not equal. Dies if the
/// scorer drops the warm start (the draw scores the cold value), if it warm-starts at the
/// estimate too (the estimate scores the warm value), or if `sir_class_etas` stops returning
/// per-class starts for a mixture or starts returning them for a non-mixture model.
#[test]
fn mixture_draws_warm_start_from_the_estimates_classes() {
    use crate::estimation::mixture::mixture_ofv;
    let (model, pop) = mixture_inputs();
    let init = unpack_params(&MIX_PACKED, &model.default_params);
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let fit = crate::api::fit(&model, &pop, &init, &opts).expect("mixture fit must run");
    let (options, params, etas) = scorer_inputs(&model, &fit);
    let class = sir_class_etas(&model, &pop, &params, &options).expect("mixture: Some");
    assert_eq!(class.len(), 2, "one start set per class");
    assert!(
        class.iter().all(|c| c.len() == pop.subjects.len()),
        "one start per subject per class"
    );
    let score = |p: &ModelParameters, starts: &[Vec<DVector<f64>>]| {
        sir_draw_ofv(&model, &pop, p, &params, &etas, Some(starts), &options)
            .expect("a mixture draw is always scored")
    };
    let ofv = |p: &ModelParameters, warm: Option<&[Vec<DVector<f64>>]>| {
        mixture_ofv(&model, &pop, p, &options, warm).ofv
    };

    let mut draw = params.clone();
    draw.theta[0] *= 1.02;
    let (cold, warm) = (ofv(&draw, None), ofv(&draw, Some(&class)));
    eprintln!(
        "MEASURE draw cold={cold:?} warm={warm:?} gap={:e}",
        warm - cold
    );
    assert!(cold.is_finite() && warm.is_finite(), "{cold} / {warm}");
    assert_ne!(
        cold.to_bits(),
        warm.to_bits(),
        "fixture premise: warm ≠ cold at the draw"
    );
    assert_eq!(
        score(&draw, &class).to_bits(),
        warm.to_bits(),
        "draw not scored warm"
    );

    // At the estimates the cold pass's own η̂ is a fixed point of the warm solve on macOS
    // arm64 (gap 0), so those starts cannot tell the arms apart; starts moved off it can.
    let moved: Vec<Vec<DVector<f64>>> = class
        .iter()
        .map(|c| c.iter().map(|e| e.add_scalar(0.05)).collect())
        .collect();
    let (cold, warm) = (ofv(&params, None), ofv(&params, Some(&moved)));
    eprintln!(
        "MEASURE estimate cold={cold:?} warm={warm:?} gap={:e}",
        warm - cold
    );
    assert_ne!(
        cold.to_bits(),
        warm.to_bits(),
        "fixture premise: warm ≠ cold at the estimate"
    );
    assert_eq!(
        score(&params, &moved).to_bits(),
        cold.to_bits(),
        "estimate not scored cold"
    );

    let prep = crate::api::prepare_run(
        root().join("examples/warfarin.ferx").to_str().unwrap(),
        Some(root().join("data/warfarin.csv").to_str().unwrap()),
    )
    .expect("warfarin must prepare");
    assert!(
        sir_class_etas(
            &prep.parsed.model,
            &prep.population,
            &prep.init_params,
            &options
        )
        .is_none(),
        "a non-mixture model has no class starts"
    );
}

/// #2382: an ODE+IOV model, one subject with two occasions whose observations are `scale`
/// times the typical prediction — the same model as the inner start-reject tests in
/// `inner_optimizer_iov_tests.rs`. At `scale = 1` no start is rejected; far above it every
/// start is (the guard rejects a start NLL above `max(1000, 250·n_obs)`).
fn ode_iov_start_reject_inputs(scale: f64) -> (CompiledModel, Population) {
    let model = crate::parse_model_string(
        "[parameters]\n  theta TVCL(0.2,0.001,10.0)\n  theta TVV(10.0,0.1,500.0)\n  omega ETA_CL ~ 0.09\n  omega ETA_V ~ 0.04\n  kappa KAPPA_CL ~ 0.01\n  sigma PROP_ERR ~ 0.2 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)\n  V = TVV * exp(ETA_V)\n[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL/V) * central\n[scaling]\n  y = central / V\n[error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = focei\n  iov_column = OCC\n",
    )
    .expect("parse ODE IOV");
    let obs_times = vec![1.0, 6.0, 25.0, 30.0];
    // The typical prediction: 100 mg into V = 10, CL = 0.2, re-dosed at 24 h.
    let typical = |t: f64| {
        let k = 0.2 / 10.0;
        10.0 * ((-k * t).exp()
            + if t >= 24.0 {
                (-k * (t - 24.0)).exp()
            } else {
                0.0
            })
    };
    let subject = Subject {
        obs_rgroup: Vec::new(),
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        observations: obs_times.iter().map(|&t| scale * typical(t)).collect(),
        obs_times,
        obs_raw_times: Vec::new(),
        obs_cmts: vec![1; 4],
        covariates: std::collections::HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 4],
        occasions: vec![1, 1, 2, 2],
        obs_l2: Vec::new(),
        dose_occasions: vec![1, 2],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
        simulated_mixnum: None,
    };
    let pop = Population {
        subjects: vec![subject],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    (model, pop)
}

/// The κ-only joint warm start #2374 made "informative": `η̂ = 0` with a fitted κ̂ per
/// occasion, `[η_CL, η_V, κ_occ1, κ_occ2]`.
const KAPPA_ONLY_START: [f64; 4] = [0.0, 0.0, 0.05, -0.04];

/// #2382: a draw whose warm re-solve rejects a subject at its inner start has no score.
/// The rejected subject's EBE is the placeholder (the unoptimised start, `H = 0`), which
/// `sir_draw_ebes` drops into the objective like any other EBE (it tracks no convergence),
/// so before #2382 the draw was scored at it. The start here is κ-only, the class #2374
/// made "informative" by handing IOV subjects their joint mode. The same subject from a
/// cold start (`η̂ = 0` without κ̂, which the guard exempts) is searched and scored, so
/// the start is what the gate reads.
///
/// Mutation: deleting `if n_start_rejected > 0 { return None; }` in `sir_draw_ofv` dies on
/// the `is_none` assertion.
#[test]
fn a_sir_draw_with_a_subject_rejected_at_its_inner_start_has_no_score() {
    let (model, pop) = ode_iov_start_reject_inputs(40.0);
    let params = model.default_params.clone();
    // A draw off the estimates, so `keep_certified_start` is off, as on every real draw.
    let mut draw = params.clone();
    draw.theta[0] *= 1.01;
    let options = FitOptions {
        verbose: false,
        ..FitOptions::default()
    };
    let score = |start: &[f64]| {
        let etas = vec![DVector::from_column_slice(start)];
        let (_, _, _, rejected) = sir_draw_ebes(&model, &pop, &draw, &params, &etas, &options);
        let ofv = sir_draw_ofv(&model, &pop, &draw, &params, &etas, None, &options);
        (rejected, ofv)
    };

    let (rejected, ofv) = score(&KAPPA_ONLY_START);
    assert_eq!(rejected, 1, "fixture premise: the κ-only start is rejected");
    assert!(
        ofv.is_none(),
        "#2382: a draw scored at a rejected start's placeholder ({ofv:?})"
    );

    let (rejected, ofv) = score(&[0.0, 0.0]);
    assert_eq!(rejected, 0, "a cold start is exempt from the guard");
    assert!(
        ofv.is_some_and(f64::is_finite),
        "the same subject from a cold start is scored ({ofv:?})"
    );
}

/// #2382: `run_sir_core` gives a draw with a rejected start weight 0 and says how many.
/// Three observation scales, one model: at 5.5 the subject's start NLL straddles the reject
/// threshold across the draws (18 of 40 rejected, measured on macOS arm64; the test needs
/// only `0 < k < 40`), at 40 every draw is rejected, and at 1 none is, so no warning.
/// Each text is pinned whole, so deleting a sentence of it reddens it.
///
/// Mutations: the draw loop recording `NonFiniteOfv` for a rejected start (or dropping
/// the draw silently) dies on both the warning and the all-rejected tally; dropping the
/// warning's `push` dies on the first; dropping the tally's count dies on the second.
#[test]
fn sir_gives_a_draw_with_a_rejected_start_weight_zero_and_counts_it() {
    let run = |scale: f64| {
        let (model, pop) = ode_iov_start_reject_inputs(scale);
        let params = model.default_params.clone();
        let n = crate::estimation::parameterization::pack_params(&params).len();
        let cov = DMatrix::from_diagonal_element(n, n, 0.01);
        let etas = vec![DVector::from_column_slice(&KAPPA_ONLY_START)];
        let opts = FitOptions {
            verbose: false,
            sir_samples: 40,
            sir_resamples: 20,
            sir_seed: Some(2382),
            ..FitOptions::default()
        };
        run_sir_core(&model, &pop, &params, &etas, &cov, 0.0, &opts)
    };
    let rejected_warning = |w: &[String]| -> Vec<String> {
        w.iter()
            .filter(|w| w.contains("SIR draws were given weight 0"))
            .cloned()
            .collect()
    };

    let partial = run(5.5).expect("some draws are scored");
    let w = rejected_warning(&partial.warnings);
    assert_eq!(w.len(), 1, "{:?}", partial.warnings);
    let k: usize = w[0]
        .split_once(" of ")
        .and_then(|(k, _)| k.parse().ok())
        .unwrap_or_else(|| panic!("no count in {:?}", w[0]));
    assert!(0 < k && k < 40, "fixture drift: {k} of 40 rejected");
    assert_eq!(
        w[0],
        format!(
            "{k} of 40 SIR draws were given weight 0: at their parameters the inner solve \
             rejected a subject's warm start, so the draw has no objective to score. The \
             intervals come from the remaining draws."
        )
    );

    let none = run(1.0).expect("every draw is scored");
    assert!(
        rejected_warning(&none.warnings).is_empty(),
        "no draw was rejected, so no warning: {:?}",
        none.warnings
    );

    let err = run(40.0).expect_err("no draw is scored");
    assert!(
        err.starts_with(
            "All 40 SIR samples had invalid weights (rejected: 0 out of bounds, 0 \
             non-finite theta or non-positive sigma/omega/kappa variance, 0 non-finite OFV, \
             40 with a subject rejected at its inner start, 0 cancelled)."
        ),
        "{err}"
    );
}
