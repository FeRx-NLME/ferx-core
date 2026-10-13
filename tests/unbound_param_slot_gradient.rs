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

fn warfarin() -> Population {
    read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data must load")
}

/// OFV at the initial parameters under one inner-gradient route.
fn eval_only_ofv(gradient_method: GradientMethod) -> (f64, Vec<String>) {
    let model = parse_model_string(UNBOUND_AFTER_CL).expect("model must parse");
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
    let (analytic, warnings) = eval_only_ofv(GradientMethod::Auto);
    let (fd, _) = eval_only_ofv(GradientMethod::Fd);

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

    // Measured (macOS arm64, this fixture): analytic 4980.879260375328,
    // fd 4980.879260374284 — relative difference 2.1e-13, the two inner optimizers
    // stopping at η̂ that agree to the inner convergence tolerance and nothing more.
    // The bound below is that number with ~5000x headroom.
    //
    // With the fix toggled off (both fallback sites keyed back on `pk_indices`) the
    // analytic arm returns 4991.568002512913 — a 10.69 OFV gap, 5e9 times this
    // bound. So the bound is nowhere near the loose kind that stops testing.
    let rel = (analytic - fd).abs() / fd.abs().max(1.0);
    assert!(
        rel < 1e-9,
        "analytic and FD inner gradients must agree: analytic {analytic}, fd {fd}, rel {rel}"
    );
}
