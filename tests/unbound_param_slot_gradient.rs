//! Tier-2 (#1899): an analytical model carrying an individual parameter that no
//! `pk(...)` role binds must produce the same objective on the **analytic** inner
//! gradient as on **finite differences**.
//!
//! The defect this pins was in the closed-form log-normal sensitivity fallback: it
//! keyed its `∂p/∂(θ,η)` rows by `pk_indices`, where such a name carries the
//! placeholder slot `0` — `CL`'s slot — so the unbound name's row overwrote `CL`'s
//! and FOCE/FOCEI ran on a gradient whose `∂f/∂η_CL` was identically zero, with the
//! subject still reported as analytic-supported.
//!
//! `gradient = fd` never read that row map, so the two routes are genuinely
//! independent here, and their agreement is the end-to-end oracle. There is no
//! NONMEM equivalent: the quantity is which gradient row a PK slot maps to for a
//! name that influences no prediction, and NONMEM's `$PK` has no slot map — every
//! `$PK` variable is just a variable, so there is no control stream that spells the
//! difference. The exact-derivative oracle (central FD of the production predictor)
//! lives in the Tier-1 `an_unbound_lognormal_parameter_does_not_steal_cls_gradient_row`.
//!
//! Eval-only (`outer_maxiter = 0`), so this returns immediately.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, GradientMethod, Population};
use std::path::Path;

mod common;

/// Warfarin 1-cpt oral with a **literal** `ka` — which is what makes the
/// Dual2-over-program path decline, putting the subject on the closed-form
/// log-normal fallback under test — plus `S`, an individual parameter with its own
/// η that nothing binds and nothing reads. `S` is declared **after** `CL`, the
/// order in which the old `seed_dim_from_slots` (last row per slot wins) let it
/// take `CL`'s row.
const UNBOUND_AFTER_CL: &str = r"
[parameters]
  theta THCL(0.134, 0.001, 10.0)
  theta THV(8.11, 0.1, 500.0)
  theta THS(3.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  omega ETA_S ~ 0.25
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = THCL * exp(ETA_CL)
  V = THV * exp(ETA_V)
  S = THS * exp(ETA_S)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=1.0)
[error_model]
  DV ~ proportional(PROP_ERR)
";

/// The same model with `KEL = CL / V` in place of `S`. `KEL` is not
/// `g(θ, cov)·exp(η)`, so the #1669 measured log-normal flag is false and **every**
/// subject routes to finite differences — which makes the FD-fallback warning fire.
/// It is here to prove the `fallback.is_empty()` assertion below can observe an
/// all-FD population at all: that warning is only emitted for a mixed population
/// *or* a model that reports analytic, and a guard nothing can trip is not a guard.
const DERIVED_INTERMEDIATE: &str = r"
[parameters]
  theta THCL(0.134, 0.001, 10.0)
  theta THV(8.11, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL = THCL * exp(ETA_CL)
  V = THV * exp(ETA_V)
  KEL = CL / V
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=1.0)
[error_model]
  DV ~ proportional(PROP_ERR)
";

fn warfarin() -> Population {
    read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data must load")
}

/// OFV at the initial parameters under one inner-gradient route.
fn eval_only_ofv(src: &str, gradient_method: GradientMethod) -> (f64, Vec<String>) {
    let model = parse_model_string(src).expect("model must parse");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        outer_maxiter: 0,
        run_covariance_step: false,
        gradient_method,
        threads: Some(2),
        ..FitOptions::default()
    };
    let r = fit(&model, &warfarin(), &model.default_params, &opts).expect("eval-only fit");
    assert!(
        r.ofv.is_finite(),
        "{gradient_method:?}: OFV must be finite, got {}",
        r.ofv
    );
    (r.ofv, r.warnings.clone())
}

#[test]
fn an_unbound_parameters_slot_does_not_move_the_objective() {
    let (analytic, warnings) = eval_only_ofv(UNBOUND_AFTER_CL, GradientMethod::Auto);
    let (fd, _) = eval_only_ofv(UNBOUND_AFTER_CL, GradientMethod::Fd);

    // The analytic arm must actually have run the analytic route; if every subject
    // fell back to finite differences the comparison is FD against FD and cannot
    // see the defect at all (AGENTS.md: ask which engine each fixture ran on).
    let fallback: Vec<&String> = warnings
        .iter()
        .filter(|w| w.contains("finite-difference inner gradient"))
        .collect();
    assert!(
        fallback.is_empty(),
        "the Auto arm must stay on the analytic inner gradient: {fallback:?}"
    );
    // The other side of that gate, in the same test: swap the unbound parameter for a
    // derived one and every subject routes to FD, so the warning this filter looks for
    // must appear. Without this the guard above is satisfied by a warning the fit never
    // emits, by a reworded message, or by an eval-only fit that skips the check.
    let (_, fd_warnings) = eval_only_ofv(DERIVED_INTERMEDIATE, GradientMethod::Auto);
    assert!(
        fd_warnings
            .iter()
            .any(|w| w.contains("finite-difference inner gradient")),
        "an all-FD population must report it, or the guard above is dead: {fd_warnings:?}"
    );

    // Measured on the reference platform — `tools/linux-test.sh --jobs 2 -- test
    // --test unbound_param_slot_gradient --no-default-features --features ci`,
    // `ferx-platform: Linux/aarch64 glibc 2.41`:
    //
    //   analytic = 4980.879260375328   fd = 4980.879260375087
    //   relative difference 4.8e-14
    //
    // The two inner optimizers stop at η̂ that agree to the inner convergence
    // tolerance and nothing more. macOS arm64 gives the same analytic value to
    // every digit and 2.1e-13 on the FD arm; the bound below is the Linux number
    // with ~2e4 headroom, which also clears macOS by ~5e3.
    //
    // With the fix toggled off (both fallback sites keyed back on `pk_indices`) the
    // analytic arm returns 4991.568002512913 — a 10.69 OFV gap, relative 2.1e-3,
    // i.e. 2e6 times this bound. So the bound is nowhere near the loose kind that
    // stops testing. It is in any case a *relative* agreement between two routes in
    // one process rather than an absolute fit band, so the OS math library shifts
    // both arms together.

    // Printed so the realised agreement can be read off a `--nocapture` run on the
    // reference platform (`tools/linux-test.sh`, or `gh workflow run slow-tests.yml
    // -f nocapture=true`) rather than re-derived from a local one.
    eprintln!("gradient-route OFV: analytic={analytic:.12} fd={fd:.12}");
    let rel = (analytic - fd).abs() / fd.abs().max(1.0);
    assert!(
        rel < 1e-9,
        "analytic and FD inner gradients must agree: analytic {analytic}, fd {fd}, rel {rel}"
    );
}
