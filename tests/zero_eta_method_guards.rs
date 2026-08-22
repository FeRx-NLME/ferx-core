//! End-to-end guards for fixed-effects-only models (`n_eta = 0`), #1006 / #1007.
//!
//! The unit tests in `src/` cover the two decision functions in isolation
//! (`check_model_options`, `keep_gn_zero_eta_warning`). What they cannot cover is
//! the **wiring**: whether a diagnostic those functions produce actually reaches
//! `FitResult.warnings` / the `Err`. Both halves of that wiring are easy to break
//! silently —
//!
//!   * `fit_inner` consumes `check_model_options` through `first_error`, which
//!     drops warning-severity diagnostics unless they are pushed on explicitly;
//!   * the GN post-fit warning is emitted inside `run_foce_gn`, which cannot see
//!     the method chain, so its chain exemption is applied by `fit_inner` instead.
//!
//! Tier 2: each fit runs a 2-iteration budget and returns immediately, or returns
//! `Err` before any stage runs. No convergence loops.

use ferx_core::*;
use std::path::Path;

/// The `n_eta = 0` anchor used throughout #989 / #1002 / #1006.
fn pooled() -> (CompiledModel, Population) {
    let model = ferx_core::parser::model_parser::parse_model_file(Path::new(
        "examples/one_cpt_iv_pooled.ferx",
    ))
    .expect("pooled example parses");
    assert_eq!(model.n_eta, 0, "fixture must be fixed-effects-only");
    let pop = read_nonmem_csv(Path::new("data/one_cpt_iv.csv"), None, None).expect("data loads");
    (model, pop)
}

/// A budget small enough that the fit returns immediately and `gn` cannot converge.
fn budgeted(opts: FitOptions) -> FitOptions {
    FitOptions {
        outer_maxiter: 2,
        run_covariance_step: false,
        verbose: false,
        ..opts
    }
}

const START_SENSITIVE: &str = "start-sensitive"; // W_GN_NO_RANDOM_EFFECTS
const BADLY_WRONG: &str = "may be badly wrong"; // the GN post-fit warning
const GENERIC_GN: &str = "max iterations reached without convergence";

#[test]
fn standalone_gn_at_zero_eta_surfaces_both_diagnostics_from_fit() {
    let (model, pop) = pooled();
    let opts = budgeted(FitOptions {
        method: EstimationMethod::FoceGn,
        ..Default::default()
    });
    let res = fit(&model, &pop, &model.default_params, &opts).expect("fit returns");
    assert!(!res.converged, "a 2-iteration GN budget cannot converge");
    // `check_model_options` only produces this as a *warning*, and `first_error`
    // discards warnings — so this asserts the explicit hand-off in `fit_inner`,
    // not just that the diagnostic was produced.
    assert!(
        res.warnings.iter().any(|w| w.contains(START_SENSITIVE)),
        "W_GN_NO_RANDOM_EFFECTS must reach fit(), not just `ferx check`: {:?}",
        res.warnings
    );
    assert!(
        res.warnings.iter().any(|w| w.contains(BADLY_WRONG)),
        "the post-fit #1006 warning must reach fit(): {:?}",
        res.warnings
    );
}

#[test]
fn the_gn_remedy_is_stated_exactly_once() {
    // The check-time warning owns the remedy and the post-fit one owns the
    // outcome. Before the split a single `gn` run said "use gn_hybrid or focei"
    // twice. Pinned from both sides: the advice must appear (it is the useful
    // half) and must appear once, so neither restoring it to the post-fit
    // message nor dropping it from the check passes.
    let (model, pop) = pooled();
    let opts = budgeted(FitOptions {
        method: EstimationMethod::FoceGn,
        ..Default::default()
    });
    let res = fit(&model, &pop, &model.default_params, &opts).expect("fit returns");
    let with_remedy: Vec<&String> = res
        .warnings
        .iter()
        .filter(|w| w.contains("gn_hybrid"))
        .collect();
    assert_eq!(
        with_remedy.len(),
        1,
        "the gn remedy must be stated exactly once: {:?}",
        res.warnings
    );
    assert!(
        with_remedy[0].contains(START_SENSITIVE) && with_remedy[0].contains("sigma start"),
        "the remedy belongs to W_GN_NO_RANDOM_EFFECTS, in full: {}",
        with_remedy[0]
    );
    let post_fit = res
        .warnings
        .iter()
        .find(|w| w.contains(BADLY_WRONG))
        .expect("post-fit warning present");
    assert!(
        !post_fit.contains("gn_hybrid") && !post_fit.contains("sigma start"),
        "the post-fit warning states the outcome only: {post_fit}"
    );
}

#[test]
fn a_gn_focei_chain_at_zero_eta_drops_the_badly_wrong_claim() {
    // `methods = [gn, focei]` is `gn_hybrid` spelled out: the FOCEI stage
    // re-optimises the GN result, so neither #1006 diagnostic applies. The GN
    // stage still reports its own non-convergence — only the targeted string is
    // suppressed, not every warning the stage produced.
    let (model, pop) = pooled();
    let opts = budgeted(FitOptions {
        methods: vec![EstimationMethod::FoceGn, EstimationMethod::FoceI],
        ..Default::default()
    });
    let res = fit(&model, &pop, &model.default_params, &opts).expect("fit returns");
    assert!(
        !res.warnings.iter().any(|w| w.contains(BADLY_WRONG)),
        "a polished GN stage must not claim the result may be badly wrong: {:?}",
        res.warnings
    );
    assert!(
        !res.warnings.iter().any(|w| w.contains(START_SENSITIVE)),
        "a polished GN stage must not get the check-time warning either: {:?}",
        res.warnings
    );
    assert!(
        res.warnings.iter().any(|w| w.contains(GENERIC_GN)),
        "the GN stage's own warnings must still come through: {:?}",
        res.warnings
    );
}

#[test]
fn an_imp_chain_at_zero_eta_fails_before_the_focei_stage_runs() {
    // #1007: `[focei, imp]` used to run the whole FOCEI stage before the IMP
    // stage errored at run time. `n_iterations` is unobservable on an `Err`, so
    // the fail-fast is pinned by the message: the run-time guards word this
    // differently ("Importance sampling requires…"), so seeing the
    // `check_model_options` wording proves the fit never reached the IMP stage.
    let (model, pop) = pooled();
    let opts = FitOptions {
        methods: vec![EstimationMethod::FoceI, EstimationMethod::Imp],
        ..Default::default()
    };
    let err = fit(&model, &pop, &model.default_params, &opts).expect_err("must be rejected");
    assert!(
        err.contains("method = imp requires at least one random effect"),
        "expected the up-front check_model_options message, got: {err}"
    );
}

#[test]
fn the_two_no_random_effect_messages_are_unchanged() {
    // `E_SAEM_NO_RANDOM_EFFECTS` is documented as stable and its message is the
    // text #1002 shipped; the #1007 sibling shares the surrounding sentence and
    // varies only the per-method clause. Both are asserted in full so a future
    // refactor of the shared table cannot quietly hand one method the other's
    // rationale — the codes alone would still look right.
    let (model, _) = pooled();
    for (method, code, expected) in [
        (
            EstimationMethod::Saem,
            "E_SAEM_NO_RANDOM_EFFECTS",
            "method = saem requires at least one random effect (n_eta = 0). \
             SAEM is an EM over the random effects, so with none declared its \
             E-step is empty and it only approaches the objective that FOCE/FOCEI \
             minimise exactly. Use method = foce, focei, or laplace for a \
             fixed-effects-only (naive-pooled) model.",
        ),
        (
            EstimationMethod::Imp,
            "E_METHOD_NO_RANDOM_EFFECTS",
            "method = imp requires at least one random effect (n_eta = 0). \
             With no random effects the marginal likelihood is just the \
             observation likelihood, which FOCE/FOCEI minimise exactly. \
             Use method = foce, focei, or laplace for a fixed-effects-only \
             (naive-pooled) model.",
        ),
    ] {
        let opts = FitOptions {
            method,
            ..Default::default()
        };
        let diags = check_model_options(&model, &opts);
        let hit = diags
            .iter()
            .find(|d| d.code == code)
            .unwrap_or_else(|| panic!("{code} not emitted: {diags:?}"));
        assert_eq!(hit.message, expected, "{code} message text changed");
    }
}
