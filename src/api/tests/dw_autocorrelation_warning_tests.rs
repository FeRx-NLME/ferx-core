//! Exactly what the pooled-IWRES autocorrelation warning says (#1285, #1350
//! row 10a).
//!
//! The assertions are **exact string equality**, not `contains`. The sentence
//! removed in this change — `" For ODE models, SDE process noise may also
//! help."` — is a recommendation, so a `!msg.contains("SDE")` assertion would
//! pass on any rewording that still sends the user into `[diffusion]` for
//! residual autocorrelation, which the EKF cannot supply while it never
//! corrects the state mean with the observed data. Nothing tested the suffix at
//! all, which is how it outlived the filing of #1285.
//!
//! Classification is *not* re-asserted here: `crate::types::classify_warning`
//! keys on "autocorrelation" / "durbin", and `types_tests.rs` already pins both
//! of these messages to the `dw_autocorrelation` category. A second copy of
//! that assertion here would cover for the first one.
//!
//! The string tests below pin the **helper**. What `fit_inner` actually pushes
//! is a separate property, and pinning only the helper left the pre-PR
//! user-visible behaviour reachable: re-appending the sentence *at the call
//! site* killed none of them (measured on review of `1d816920`). That is what
//! `fit_pushes_exactly_the_helpers_message` below exists for.

use super::*;
use crate::parser::model_parser::parse_model_string;
use std::collections::HashMap;

/// Positive autocorrelation, character for character.
///
/// Mutations that must redden this: re-appending the SDE sentence inside the
/// helper (the state of the tree before #1350 row 10a); rewording the three
/// remedies the warning does still name. It does **not** see a sentence
/// appended by a caller — `fit_pushes_exactly_the_helpers_message` is the test
/// for that.
#[test]
fn positive_autocorrelation_message_is_exact() {
    let msg = dw_autocorrelation_warning(1.20, &DwAdvice::Compartmental)
        .expect("DW = 1.20 is below the 1.5 threshold");
    assert_eq!(
        msg,
        "Positive IWRES autocorrelation detected (Durbin-Watson = 1.20). \
         Structural model may be missing dynamics. Consider a transit \
         absorption model, additional compartment, or IOV on ka/F."
    );
}

/// Negative autocorrelation, character for character. Untouched by #1285 — the
/// SDE suffix was only ever appended to the positive branch — and pinned here
/// so a future edit to the shared helper cannot quietly move it either.
#[test]
fn negative_autocorrelation_message_is_exact() {
    let msg = dw_autocorrelation_warning(2.80, &DwAdvice::Compartmental)
        .expect("DW = 2.80 is above the 2.5 threshold");
    assert_eq!(
        msg,
        "Negative IWRES autocorrelation detected (Durbin-Watson = 2.80). \
         Possible over-parameterization or misspecified error model."
    );
}

/// The quiet band, edges included: `< 1.5` and `> 2.5` are strict, so 1.5 and
/// 2.5 themselves warn about nothing.
///
/// Mutation that must redden this: swapping the two thresholds (`< 2.5` /
/// `> 1.5`), which turns every DW in the band into a warning.
#[test]
fn no_warning_inside_the_band() {
    for dw in [1.5, 2.0, 2.5] {
        assert_eq!(
            dw_autocorrelation_warning(dw, &DwAdvice::Compartmental),
            None,
            "DW = {dw} is inside the quiet band"
        );
    }
}

/// `dw_statistic` is `NaN` when no subject has two finite IWRES values, and the
/// comparisons above are both false on `NaN` — but relying on that leaves the
/// guard deletable, so the non-finite cases are asserted directly.
#[test]
fn non_finite_dw_never_warns() {
    for dw in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            dw_autocorrelation_warning(dw, &DwAdvice::Compartmental),
            None,
            "a non-finite DW ({dw}) is a missing statistic, not a diagnosis"
        );
    }
}

/// A one-compartment ODE model whose fixed effects cannot follow a
/// two-compartment truth: `CL/V = 0.1` against a biexponential.
///
/// It carries an `omega`, and that is measured rather than argued. An earlier
/// version dropped it on the theory that a random effect on `CL` would absorb
/// the drift through the EBE search; it cannot, because a scalar on `CL` cannot
/// bend a mono-exponential into a bi-exponential. Durbin-Watson on this fixture
/// (the two subjects of `assert_fit_pushes_only_the_helpers_message`), one
/// variable changed at a time — the same to four digits on native macOS arm64 and
/// on Linux aarch64 (`tools/linux-test.sh`):
///
/// | variant | `outer_maxiter: 1` | converged | converged ω |
/// |---|---|---|---|
/// | no `omega` | 0.1448 | 0.1257 | — |
/// | `omega ETA_CL ~ 0.09` (ODE arm) | 0.2394 | 0.2439 | 0.068 |
/// | the same, `[diffusion]` arm | 0.2468 | 0.2468 | 0.069 |
///
/// Those are the classic `÷ Σ e²` values. Under the pair-normalised statistic
/// (#2798) the two arms the test runs read 0.2660 (ODE) and 0.2776 (`[diffusion]`)
/// at `outer_maxiter: 1` (native macOS arm64; the `[DW]` line the helper prints).
///
/// Worst case 0.28 against a bound of 1.5, so the `omega` is free. Keeping it
/// matters for what the test can *see*: with `n_eta = 0` the fixture cannot
/// observe a suffix gated on `model.n_eta > 0`, and it sits off the analytic
/// inner-gradient path (see #1432).
///
/// The ω has to be *identifiable*, not merely present: the subjects differ in
/// their elimination rate (see `biexponential_subject`), so `ETA_CL` has an
/// interior optimum. An earlier fixture differed only in an amount scale, which
/// `CL` cannot express, so ω's optimum was 0 (η shrinkage 98 %) and whether the
/// fit stopped just short of the packed-coordinate guard or on it depended on the
/// platform: on native macOS arm64 the `[diffusion]` arm — whose FD fit is
/// already converged at `outer_maxiter: 1` — reached the guard and emitted a
/// collapse warning the allow-list rightly does not account for, while Linux
/// stopped short and passed.
///
/// `TVV` is `FIX`ed because the observation is the compartment **amount**, so
/// `V` enters only through `CL / V` — free, it is a flat direction rather than
/// an estimated parameter.
fn one_cpt_model() -> CompiledModel {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0) FIX
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[error_model]
  DV ~ proportional(PROP)
"#;
    parse_model_string(src).expect("parse")
}

/// The same model with a `[diffusion]` block, so the pair differs in whether
/// `model.is_sde()` holds — and in the `fd` gradient that block forces (below).
///
/// Without this arm the test cannot see a recommendation gated on `is_sde()` —
/// which is the population that must least of all be told to add process noise.
/// `gradient_method = fd` because the EKF path has no analytic-sensitivity
/// provider.
///
/// The diffusion is `FIX`ed: what the arm needs is `is_sde()`, not a diffusion
/// estimate, and an estimated one competes with ω and σ for the same misfit —
/// measured, it ran to 9 or 59 in variants of this fixture and drove σ to its floor
/// (an `EPS shrinkage` warning) or ω to its guard.
fn one_cpt_sde_model() -> CompiledModel {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0) FIX
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL / V) * central
[diffusion]
  central ~ 0.01 FIX
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  gradient_method = fd
"#;
    parse_model_string(src).expect("parse")
}

/// Observations from a two-compartment truth,
/// `scale·100·(0.6·e^{-0.5·rate·t} + 0.4·e^{-0.05·rate·t})`. Against the model's
/// single `e^{-(CL/V)t}` the residuals are a smooth U — strongly positively
/// autocorrelated, which is the `DW < 1.5` branch. `rate` differs between the two
/// subjects so that `ETA_CL` is identifiable (see `one_cpt_model`).
fn biexponential_subject(id: &str, scale: f64, rate: f64) -> Subject {
    let obs_times: Vec<f64> = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0];
    let observations: Vec<f64> = obs_times
        .iter()
        .map(|t| scale * 100.0 * (0.6 * (-0.5 * rate * t).exp() + 0.4 * (-0.05 * rate * t).exp()))
        .collect();
    let n = obs_times.len();
    Subject {
        obs_rgroup: Vec::new(),
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![1; n],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cmt_toggles: Vec::new(),
        cens: vec![0; n],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
        simulated_mixnum: None,
    }
}

/// What `fit_inner` pushes is the helper's message, unmodified — and it pushes
/// nothing else about autocorrelation.
///
/// The four tests above pin `dw_autocorrelation_warning` in isolation, and that
/// left the pre-PR user-visible behaviour reachable through the *caller*:
/// re-appending `" For ODE models, SDE process noise may also help."` to `msg`
/// inside `fit_inner` passed the entire `ferx-core/ci` lib suite (measured on
/// review of `1d816920`). Nothing under `tests/` or `crates/` looked at the text
/// either.
///
/// The first version of this test filtered the warnings on
/// `contains("autocorrelation")` — the helper's own vocabulary — so the same
/// recommendation pushed as its **own** entry still passed (measured on review
/// of `c7c38c13`). Hence the assertion below is over the *whole*
/// `result.warnings`: every entry must be either the helper's message or one of
/// the warnings this fixture is entitled to emit, so a suffix, a prefix, a
/// second `push`, and a reworded copy all fail.
///
/// The fixture is an **ODE** model on purpose. The removed suffix was gated on
/// `model.ode_spec.is_some()`, so an analytic model cannot observe its return,
/// and it was appended only below 1.5 — so the fixture has to land on that side
/// of the gate. Both facts are asserted rather than assumed: a fixture that
/// drifted into the quiet band would otherwise leave this test passing on an
/// empty search.
/// Every warning either fixture is *entitled* to emit, as (opening, closing)
/// pairs. Both ends are pinned on purpose: with prefixes alone, appending the
/// recommendation to an already-accounted warning — the budget notice, say —
/// stays accounted and the test passes (measured; that is how this list came to
/// be anchored at both ends).
///
/// There is deliberately no `High ETA shrinkage` entry. With an identifiable ω
/// (see `one_cpt_model`) neither arm emits one, and allowing it would let a
/// regression back to the collapsing-ω fixture pass unaccounted.
///
/// The thread notice opens with a machine-dependent count, so its "prefix" is
/// `classify_warning`'s own key, matched anywhere in the string.
const ACCOUNTED: &[(&str, &str)] = &[
    (
        "No estimation method was specified",
        "to silence this warning.",
    ),
    (
        "Outer optimization hit the evaluation budget",
        "increase maxiter for a tighter fit.",
    ),
    ("Outer optimization did not converge", "did not converge"),
    (
        "The outer optimizer stopped without converging (#2713).",
        "such as `if`, `abs` or `max`.",
    ),
    ("threads configured", "(no speed benefit beyond n_subjects)"),
    (
        "Stochastic differential equations ([diffusion]",
        "See the Feature Maturity and SDE pages in the documentation.",
    ),
    // The SS non-convergence sink is one set for the whole process, and `fit()` drains it
    // into `FitResult.warnings`, so a concurrently running lib test that caps an SS
    // equilibration (a Michaelis–Menten model dosed at steady state) lands its warning
    // here; no lib test can assert the sink empty (`dosing.rs`, #1289). This fixture has no
    // SS dose and never writes one. Measured: 1 failure in 23 runs of the `mixture optimizer
    // auto` filter on `main` at 1a770c5e, both messages from another test.
    (
        "Steady-state (SS=1) equilibration did not converge within",
        "The returned SS trough is unreliable.",
    ),
    (
        "Steady-state (SS=1) equilibration reached the",
        "simulations for such a model may be biased low.",
    ),
];

/// Fit `model` and assert that the only thing `fit_inner` says about
/// autocorrelation is the helper's message, and that it says nothing else at all
/// beyond the warnings above.
fn assert_fit_pushes_only_the_helpers_message(model: &CompiledModel, label: &str) -> String {
    let pop = population(vec![
        biexponential_subject("1", 1.0, 1.0),
        biexponential_subject("2", 1.1, 2.5),
    ]);
    assert_fit_pushes_only_the_helpers_message_on(model, &pop, FitOptions::default(), label)
}

fn population(subjects: Vec<Subject>) -> Population {
    Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// [`assert_fit_pushes_only_the_helpers_message`] on a given population, fitted
/// under `base` (one outer iteration, no covariance step); the expected message is
/// the helper's for the advice `dw_advice` resolves under those same options
/// (#2798). Returns that message so a caller can pin which advice reached it.
fn assert_fit_pushes_only_the_helpers_message_on(
    model: &CompiledModel,
    pop: &Population,
    base: FitOptions,
    label: &str,
) -> String {
    // One outer iteration: the statistic comes from the post-fit residual pass,
    // and the misfit is structural, so there is nothing to gain by converging.
    let opts = FitOptions {
        outer_maxiter: 1,
        run_covariance_step: false,
        ..base
    };

    let result = fit(model, pop, &model.default_params, &opts).expect("fit should succeed");
    eprintln!("[{label}] DW {:.4}", result.dw_statistic);

    assert!(
        result.dw_statistic.is_finite() && result.dw_statistic < 1.5,
        "[{label}] fixture must land on the branch the SDE sentence was appended to; DW = {}",
        result.dw_statistic
    );

    let mut notes = Vec::new();
    let prepared = fitted_population(
        model,
        &opts.iov_occasion,
        opts.iov_column.is_some(),
        std::borrow::Cow::Borrowed(pop),
        &mut notes,
    );
    let expected =
        dw_autocorrelation_warning(result.dw_statistic, &dw_advice(model, &opts, &prepared))
            .expect("a DW below 1.5 must produce a message");

    let accounted = |w: &str| {
        w == expected
            || ACCOUNTED
                .iter()
                .any(|(open, close)| w.contains(open) && w.ends_with(close))
    };

    // The gate can fire, in both of the shapes that have slipped past a weaker
    // version of this test: the recommendation as its own entry, and the
    // recommendation appended to a warning that *is* accounted for.
    let removed = "For ODE models, SDE process noise may also help.";
    assert!(
        !accounted(removed),
        "[{label}] the allow-list must not absorb the removed recommendation"
    );
    for w in &result.warnings {
        let tampered = format!("{w} {removed}");
        assert!(
            !accounted(&tampered),
            "[{label}] appending to an accounted warning must not stay accounted: {w}"
        );
    }

    let unaccounted: Vec<&String> = result.warnings.iter().filter(|w| !accounted(w)).collect();
    assert!(
        unaccounted.is_empty(),
        "[{label}] fit() emitted warnings this fixture does not account for: {unaccounted:#?}"
    );

    let dw_entries: Vec<&String> = result
        .warnings
        .iter()
        .filter(|w| {
            crate::types::classify_warning(w).category
                == crate::types::WarningCode::DwAutocorrelation
        })
        .collect();
    assert_eq!(
        dw_entries.len(),
        1,
        "[{label}] expected exactly one dw_autocorrelation warning, got {dw_entries:#?}"
    );
    assert_eq!(
        *dw_entries[0], expected,
        "[{label}] fit_inner must push the helper's message unmodified"
    );
    expected
}

#[test]
fn fit_pushes_exactly_the_helpers_message() {
    // Two arms differing in exactly one thing: whether the model is an SDE.
    // A recommendation re-introduced behind `model.is_sde()` is invisible to a
    // plain-ODE fixture (measured), and `[diffusion]` users are the population
    // that must least of all be told to add process noise.
    let ode = one_cpt_model();
    assert!(
        ode.ode_spec.is_some(),
        "the removed suffix was gated on `ode_spec`; an analytic fixture cannot observe it"
    );
    assert!(!ode.is_sde(), "the ODE arm must not be an SDE model");
    let pushed = assert_fit_pushes_only_the_helpers_message(&ode, "ODE");
    assert!(
        pushed.ends_with(
            "Consider a transit absorption model, additional compartment, or IOV on ka/F."
        ),
        "an ODE model keeps the structural advice (#2798): {pushed}"
    );

    let sde = one_cpt_sde_model();
    assert!(sde.ode_spec.is_some());
    assert!(sde.is_sde(), "the SDE arm must carry a [diffusion] block");
    let _ = assert_fit_pushes_only_the_helpers_message(&sde, "SDE");
}

/// A compartment-free straight line `y = A + B·TIME` fitted to parabolas: the
/// residuals are a smooth U, the `DW < 1.5` branch. `A` carries the η, and the
/// subjects differ in their offset so it is identifiable. No doses.
fn line_model() -> CompiledModel {
    let src = r#"
[parameters]
  theta TVA(10.0, 0.1, 500.0)
  theta TVB(0.5, -50.0, 50.0)
  omega ETA_A ~ 0.09
  sigma ADD ~ 1.0
[individual_parameters]
  A = TVA * exp(ETA_A)
  B = TVB
[structural_model]
  y = A + B * TIME
[error_model]
  DV ~ additive(ADD)
"#;
    parse_model_string(src).expect("parse")
}

fn parabola_subject(id: &str, offset: f64) -> Subject {
    let obs_times: Vec<f64> = (0..10).map(|i| 2.0 * f64::from(i)).collect();
    let observations: Vec<f64> = obs_times
        .iter()
        .map(|t| offset + 0.5 * t + 0.1 * (t - 9.0) * (t - 9.0))
        .collect();
    let n = obs_times.len();
    let mut s = biexponential_subject(id, 1.0, 1.0);
    s.doses = Vec::new();
    s.obs_cmts = vec![1; n];
    s.cens = vec![0; n];
    s.obs_times = obs_times;
    s.observations = observations;
    s
}

/// What `fit_inner` pushes for a compartment-free model is the helper's
/// **compartment-free** message (#2798), unmodified, and nothing else.
///
/// Mutations that must redden this: passing `DwAdvice::Compartmental`
/// unconditionally at the call site; re-appending the PK remedies there.
#[test]
fn fit_pushes_the_compartment_free_message_for_an_algebraic_model() {
    let model = line_model();
    assert!(model.is_algebraic(), "the fixture must be compartment-free");
    let pop = population(vec![
        parabola_subject("1", 10.0),
        parabola_subject("2", 14.0),
        parabola_subject("3", 7.0),
    ]);
    let pushed = assert_fit_pushes_only_the_helpers_message_on(
        &model,
        &pop,
        FitOptions::default(),
        "algebraic",
    );
    assert!(
        pushed.contains("correlation ar1(RHO) by COL") && !pushed.contains("transit"),
        "the compartment-free advice must reach the user: {pushed}"
    );
}

/// The grouping reaches the pushed message through the **fit's** options: the
/// same line model with `iov_column = OCC` in `[fit_options]`, fitted under the
/// parsed options, pushes `by OCC` (#2815 review round 1, row 2). `dw_advice`
/// takes `options` because the column a fit groups by is the caller's
/// (#1783); every other test calls it directly, and the end-to-end fixture above
/// declares no grouping, so its advice is the same whatever options reach it.
///
/// Each subject is two **parallel** occasions (two parabolas, the clock restarting
/// at the second), since the advice names the column only where the statistic
/// split at it (#2829).
///
/// Mutation that must redden this: `dw_advice(model, &FitOptions::default(), ..)`
/// at the call site in `fit_inner`.
#[test]
fn fit_pushes_the_grouping_its_options_carry() {
    let parsed = crate::parser::model_parser::parse_full_model(
        "[parameters]\n  theta TVA(10.0, 0.1, 500.0)\n  theta TVB(0.5, -50.0, 50.0)\n  \
         omega ETA_A ~ 0.09\n  sigma ADD ~ 1.0\n[individual_parameters]\n  \
         A = TVA * exp(ETA_A)\n  B = TVB\n[structural_model]\n  y = A + B * TIME\n\
         [error_model]\n  DV ~ additive(ADD)\n[fit_options]\n  iov_column = OCC\n",
    )
    .expect("parse");
    assert_eq!(parsed.fit_options.iov_column.as_deref(), Some("OCC"));
    assert!(
        parsed.model.is_algebraic(),
        "the fixture must be compartment-free"
    );
    let mut subjects = vec![
        parabola_subject("1", 10.0),
        parabola_subject("2", 14.0),
        parabola_subject("3", 7.0),
    ];
    for s in &mut subjects {
        let n = s.obs_times.len();
        s.obs_times = [s.obs_times.clone(), s.obs_times.clone()].concat();
        let second: Vec<f64> = s.observations.iter().map(|y| y + 3.0).collect();
        s.observations.extend(second);
        s.obs_cmts = vec![1; 2 * n];
        s.cens = vec![0; 2 * n];
        // What the reader writes for `iov_column = OCC`.
        s.occasions = [vec![1; n], vec![2; n]].concat();
        assert!(!crate::stats::residual_error::occasions_run_in_time(s));
    }
    let mut pop = population(subjects);
    pop.covariate_names = vec!["OCC".into()];
    let pushed = assert_fit_pushes_only_the_helpers_message_on(
        &parsed.model,
        &pop,
        parsed.fit_options.clone(),
        "algebraic, iov_column = OCC",
    );
    assert!(
        pushed.contains("correlation ar1(RHO) by OCC`") && !pushed.contains("by COL"),
        "the options' grouping must reach the user: {pushed}"
    );
}

// ── #1081: the decorrelated statistic under a `correlation` line ──────────────────

const LINE: &str = "correlation cs(RHO) by ARM";

/// The positive message under `cs`, character for character: where the statistic
/// was taken, that the line does not explain it, and the `ar1` advice.
#[test]
fn decorrelated_positive_message_under_cs_is_exact() {
    let msg =
        dw_autocorrelation_warning_decorrelated(1.43, LINE, crate::types::ResidualSerialKind::Cs)
            .expect("below 1.5");
    assert_eq!(
        msg,
        "Positive autocorrelation remains in the residuals decorrelated by `correlation \
         cs(RHO) by ARM` (Durbin-Watson = 1.43): the within-group correlation does not \
         explain it. Compound symmetry correlates every two rows of a block equally; if \
         the dependence decays with the distance between visits, `ar1` models that. \
         Otherwise the structural model may be missing dynamics."
    );
}

/// Under `ar1` the `ar1` advice would be circular; the message names the other two
/// causes instead.
#[test]
fn decorrelated_positive_message_under_ar1_is_exact() {
    let msg = dw_autocorrelation_warning_decorrelated(
        1.2,
        "correlation ar1(RHO) by ARM",
        crate::types::ResidualSerialKind::Ar1,
    )
    .expect("below 1.5");
    assert_eq!(
        msg,
        "Positive autocorrelation remains in the residuals decorrelated by `correlation \
         ar1(RHO) by ARM` (Durbin-Watson = 1.20): the within-group correlation does not \
         explain it. The structural model may be missing dynamics, or the grouping may \
         not cover the rows that are dependent."
    );
}

/// Under `car1` the advice names the one structure that differs from it — `ar1`,
/// which decays with the visit order rather than the time — and then the other two
/// causes. Exact, so deleting either sentence reddens it.
#[test]
fn decorrelated_positive_message_under_car1_is_exact() {
    let msg = dw_autocorrelation_warning_decorrelated(
        1.3,
        "correlation car1(RHO) by ARM",
        crate::types::ResidualSerialKind::Car1,
    )
    .expect("below 1.5");
    assert_eq!(
        msg,
        "Positive autocorrelation remains in the residuals decorrelated by `correlation \
         car1(RHO) by ARM` (Durbin-Watson = 1.30): the within-group correlation does not \
         explain it. `car1` decays the correlation with the time between rows; if it \
         follows the visit order instead, `ar1` models that. Otherwise the structural \
         model may be missing dynamics, or the grouping may not cover the rows that are \
         dependent."
    );
}

#[test]
fn decorrelated_negative_message_is_exact() {
    let msg =
        dw_autocorrelation_warning_decorrelated(2.7, LINE, crate::types::ResidualSerialKind::Cs)
            .expect("above 2.5");
    assert_eq!(
        msg,
        "Negative autocorrelation in the residuals decorrelated by `correlation cs(RHO) by \
         ARM` (Durbin-Watson = 2.70). The residual correlation may be over-estimated, or the \
         error model misspecified."
    );
}

/// The band and a non-finite statistic stay silent, as for the IWRES warning; and
/// both messages land in the `dw_autocorrelation` category.
#[test]
fn decorrelated_warning_band_and_category() {
    for dw in [1.5, 2.0, 2.5, f64::NAN, f64::INFINITY] {
        assert_eq!(
            dw_autocorrelation_warning_decorrelated(
                dw,
                LINE,
                crate::types::ResidualSerialKind::Ar1
            ),
            None,
            "dw = {dw}"
        );
    }
    for dw in [1.0, 3.0] {
        let msg =
            dw_autocorrelation_warning_decorrelated(dw, LINE, crate::types::ResidualSerialKind::Cs)
                .unwrap();
        assert_eq!(
            crate::types::classify_warning(&msg).category,
            crate::types::WarningCode::DwAutocorrelation,
            "{msg}"
        );
    }
}

/// B8 (#2798 PR B): the decorrelated `cs` advice, end to end. Before #2798 the NONMEM
/// anchor's `cs` fit (`durbin_watson_is_taken_on_decorrelated_residuals`) was the only
/// fit that reached it; the grouped, pair-normalised statistic moved that fixture to
/// 1.63, silent, so this synthetic one takes its place.
///
/// A straight line fitted to a parabola inside each arm, `correlation cs(RHO) by
/// ARM`: compound symmetry correlates the rows of an arm equally, which absorbs the
/// arm's shared offset but not the smooth U of the misfit along its visits — serial
/// dependence that decays with the lag, which is what the `cs` advice points at
/// `ar1` for. The statistic is the whitened one, split by arm: 0.678 at
/// `outer_maxiter: 1` (native macOS arm64), far below 1.5.
///
/// Mutations that must redden this: dropping the decorrelated arm at the call site
/// (the plain advice, or nothing, pushed instead); the `Cs` advice text deleted.
#[test]
fn fit_pushes_the_decorrelated_cs_advice() {
    let parsed = crate::parser::model_parser::parse_full_model(
        "[parameters]\n  theta TVA(10.0, 0.1, 500.0)\n  theta TVB(0.5, -50.0, 50.0)\n  \
         theta RHO(0.3, -0.1, 0.99)\n  omega ETA_A ~ 0.09\n  sigma ADD ~ 1.0\n\
         [individual_parameters]\n  A = TVA * exp(ETA_A)\n  B = TVB\n[structural_model]\n  \
         y = A + B * TIME\n[error_model]\n  DV ~ additive(ADD)\n  \
         correlation cs(RHO) by ARM\n[fit_options]\n  method = focei\n",
    )
    .expect("parse");
    let model = parsed.model;
    let serial = model.residual_serial.as_ref().expect("the line parses");
    assert_eq!(serial.kind, crate::types::ResidualSerialKind::Cs);
    // Two arms per subject, eight visits each, a parabola around a per-arm offset.
    let arm_subject = |id: &str, offsets: [f64; 2]| {
        let mut s = parabola_subject(id, 0.0);
        let times: Vec<f64> = (0..8).map(|i| 2.0 * f64::from(i)).collect();
        s.obs_times = times.iter().chain(&times).copied().collect();
        s.observations = offsets
            .iter()
            .flat_map(|&o| {
                times
                    .iter()
                    .map(move |t| o + 0.5 * t + 0.2 * (t - 7.0) * (t - 7.0))
            })
            .collect();
        let n = s.obs_times.len();
        s.obs_cmts = vec![1; n];
        s.cens = vec![0; n];
        s.obs_covariates = (0..n)
            .map(|j| HashMap::from([("ARM".to_string(), if j < 8 { 1.0 } else { 2.0 })]))
            .collect();
        s.covariates.insert("ARM".into(), 1.0);
        s
    };
    let mut pop = population(vec![
        arm_subject("1", [10.0, 13.0]),
        arm_subject("2", [14.0, 12.0]),
        arm_subject("3", [7.0, 9.0]),
    ]);
    pop.covariate_names = vec!["ARM".into()];
    let opts = FitOptions {
        outer_maxiter: 1,
        run_covariance_step: false,
        ..parsed.fit_options
    };
    let result = fit(&model, &pop, &model.default_params, &opts).expect("fit");
    eprintln!("B8 decorrelated cs DW {:.4}", result.dw_statistic);
    assert!(
        result.dw_statistic.is_finite() && result.dw_statistic < 1.5,
        "the fixture must land on the positive branch; DW = {}",
        result.dw_statistic
    );
    let expected = dw_autocorrelation_warning_decorrelated(
        result.dw_statistic,
        "correlation cs(RHO) by ARM",
        crate::types::ResidualSerialKind::Cs,
    )
    .expect("a DW below 1.5 warns");
    let dw_entries: Vec<&String> = result
        .warnings
        .iter()
        .filter(|w| {
            crate::types::classify_warning(w).category
                == crate::types::WarningCode::DwAutocorrelation
        })
        .collect();
    assert_eq!(dw_entries, vec![&expected], "{:#?}", result.warnings);
    assert!(
        expected.contains("`ar1` models that") && !expected.contains("transit"),
        "{expected}"
    );
}

// ── #2798: the positive advice names only remedies the model can take ─────────────
//
// One test per cell of the message's input space (model class × grouping), each
// asserting the `DwAdvice` the parsed model resolves to, the exact message, and the
// claims the message must and must not make — together, so one test reddens per
// mutated cell. Every message stays in the `dw_autocorrelation` category: that is
// pinned in `types_tests.rs` (`classify_warning_dw_advice_messages_are_dw`).

/// Today's positive text, which every compartmental model keeps byte for byte.
const COMPARTMENTAL: &str = "Positive IWRES autocorrelation detected (Durbin-Watson = 1.20). \
     Structural model may be missing dynamics. Consider a transit absorption model, \
     additional compartment, or IOV on ka/F.";

/// The compartment-free text when the model declares no grouping a `by` can read.
const ALGEBRAIC_UNGROUPED: &str = "Positive IWRES autocorrelation detected (Durbin-Watson = \
     1.20). Rows of one subject may share what the structural model leaves out, such as the \
     visits of one arm or occasion. A `correlation ar1(RHO) by COL` line in [error_model] \
     lets them correlate, with COL any data column that labels the group (the innermost \
     level's name when levels are nested); `car1(RHO)` decays the correlation with the time \
     between rows instead. The line needs method = foce or focei. A `level` for that \
     grouping with a `kappa … at` it can absorb the shared part instead.";

/// The compartment-free text for group `g`.
fn algebraic_grouped(g: &str) -> String {
    format!(
        "Positive IWRES autocorrelation detected (Durbin-Watson = 1.20). Rows of one `{g}` \
         group may share what the structural model leaves out. A `correlation ar1(RHO) by \
         {g}` line in [error_model] lets them correlate, decaying with the number of visits \
         between them; `car1(RHO)` decays the correlation with the time between rows \
         instead. The line needs method = foce or focei."
    )
}

#[derive(Clone, Copy, Debug)]
enum Class {
    Algebraic,
    PkAnalytic,
    Ode,
}

/// The groupings a model can declare. Level names differ from their columns
/// (`ARM` reads `ARMID`, `STUDY` reads `STUDYID`), so a name/column swap and an
/// innermost/outermost swap each change the advice.
#[derive(Clone, Copy, Debug)]
enum Grouping {
    None,
    IovColumn,
    SingleLevel,
    Nested,
    /// `iov_column = OCC` **and** `iov_occasion = time(...)`, with a kappa: the rule
    /// overrides the column, so the column must not leak into the advice.
    IovTime,
    /// The same with `iov_occasion = dose`.
    IovDose,
    /// `iov_column = OCC` and `iov_occasion = time(...)` **without** a kappa: the fit
    /// derives no occasions (`run::occasions_are_derived`), so the column stands.
    IovTimeNoKappa,
}

const GROUPINGS: [Grouping; 7] = [
    Grouping::None,
    Grouping::IovColumn,
    Grouping::SingleLevel,
    Grouping::Nested,
    Grouping::IovTime,
    Grouping::IovDose,
    Grouping::IovTimeNoKappa,
];

fn grouped_model(class: Class, grouping: Grouping) -> crate::types::ParsedModel {
    let (levels, kappas, k, fit_options) = match grouping {
        Grouping::None => ("", "", "", ""),
        Grouping::IovColumn => ("", "  kappa K ~ 0.04\n", " + K", "  iov_column = OCC\n"),
        Grouping::SingleLevel => (
            "  level ARM by ARMID in ID\n",
            "  kappa K ~ 0.04 at ARM\n",
            " + K",
            "",
        ),
        Grouping::Nested => (
            "  level STUDY by STUDYID in ID\n  level ARM by ARMID in STUDY\n",
            "  kappa K_ST ~ 0.04 at STUDY\n  kappa K ~ 0.04 at ARM\n",
            " + K_ST + K",
            "",
        ),
        Grouping::IovTime => (
            "",
            "  kappa K ~ 0.04\n",
            " + K",
            "  iov_column = OCC\n  iov_occasion = time(24, 48)\n",
        ),
        Grouping::IovDose => (
            "",
            "  kappa K ~ 0.04\n",
            " + K",
            "  iov_column = OCC\n  iov_occasion = dose\n",
        ),
        Grouping::IovTimeNoKappa => (
            "",
            "",
            "",
            "  iov_column = OCC\n  iov_occasion = time(24, 48)\n",
        ),
    };
    let (indiv, structural) = match class {
        Class::Algebraic => (
            format!("  A = TVA * exp(ETA{k})\n  B = TVB\n"),
            "  y = A + B * TIME\n".to_string(),
        ),
        Class::PkAnalytic => (
            format!("  CL = TVA * exp(ETA{k})\n  V = TVB\n"),
            "  pk one_cpt_iv(cl=CL, v=V)\n".to_string(),
        ),
        Class::Ode => (
            format!("  CL = TVA * exp(ETA{k})\n  V = TVB\n"),
            "  ode(obs_cmt=central, states=[central])\n[odes]\n  \
             d/dt(central) = -(CL / V) * central\n"
                .to_string(),
        ),
    };
    let src = format!(
        "[parameters]\n  theta TVA(1.0, 0.1, 50.0)\n  theta TVB(10.0, 1.0, 500.0)\n  \
         omega ETA ~ 0.09\n{levels}{kappas}  sigma ADD ~ 0.5\n\
         [individual_parameters]\n{indiv}[structural_model]\n{structural}\
         [error_model]\n  DV ~ additive(ADD)\n[fit_options]\n  method = focei\n{fit_options}"
    );
    let parsed = crate::parser::model_parser::parse_full_model(&src)
        .unwrap_or_else(|e| panic!("{class:?} × {grouping:?} must parse: {e}\n{src}"));
    // The predicate the advice keys on, asserted so a fixture that drifted to the
    // other class cannot pass its row vacuously.
    assert_eq!(
        parsed.model.is_algebraic(),
        matches!(class, Class::Algebraic),
        "{class:?} × {grouping:?}"
    );
    parsed
}

/// How a cell's rows are laid out in time (#2829).
#[derive(Clone, Copy, Debug)]
enum Rows {
    /// The grouping's two groups one after the other on one clock (visits).
    InTime,
    /// The second group restarts the clock (arms).
    Parallel,
}

const LAYOUTS: [Rows; 2] = [Rows::InTime, Rows::Parallel];

/// One four-row subject of a cell: two rows in each of the grouping's two groups,
/// carrying the labels the reader would write (`Subject::occasions` from
/// `iov_column`, which a single level also sets; the level columns as per-row
/// covariates for a nested model). Doses at 0, 24 and 48 give `iov_occasion = dose`
/// something to count; `time(24, 48)` cuts the in-time clock `6, 30, 54, 78` into
/// three windows.
fn cell_subject(id: &str, parsed: &crate::types::ParsedModel, rows: Rows) -> Subject {
    let mut s = biexponential_subject(id, 1.0, 1.0);
    s.doses = [0.0, 24.0, 48.0]
        .iter()
        .map(|t| DoseEvent::new(*t, 100.0, 1, 0.0, false, 0.0))
        .collect();
    s.obs_times = match rows {
        Rows::InTime => vec![6.0, 30.0, 54.0, 78.0],
        Rows::Parallel => vec![6.0, 30.0, 6.0, 30.0],
    };
    s.observations = vec![1.0; 4];
    s.obs_cmts = vec![1; 4];
    s.cens = vec![0; 4];
    if parsed.fit_options.iov_column.is_some() {
        s.occasions = vec![1, 1, 2, 2];
    }
    s.obs_covariates = [1.0, 1.0, 2.0, 2.0]
        .iter()
        .map(|arm| HashMap::from([("STUDYID".to_string(), 1.0), ("ARMID".to_string(), *arm)]))
        .collect();
    s
}

/// The population the fit would score for `subjects`: `fitted_population`, which
/// derives an `iov_occasion` rule's labels and writes a nested model's cells.
fn prepared(parsed: &crate::types::ParsedModel, subjects: Vec<Subject>) -> Population {
    let mut pop = population(subjects);
    pop.covariate_names = vec!["STUDYID".into(), "ARMID".into()];
    let mut notes = Vec::new();
    fitted_population(
        &parsed.model,
        &parsed.fit_options.iov_occasion,
        parsed.fit_options.iov_column.is_some(),
        std::borrow::Cow::Owned(pop),
        &mut notes,
    )
    .into_owned()
}

/// Whether every subject's occasions run in time, the predicate the advice and the
/// statistic share.
fn all_in_time(pop: &Population) -> bool {
    pop.subjects
        .iter()
        .all(crate::stats::residual_error::occasions_run_in_time)
}

/// The advice and positive message for one cell, on a one-subject population laid
/// out as `rows`, and whether its prepared occasions run in time.
fn cell(class: Class, grouping: Grouping, rows: Rows) -> (DwAdvice, String, bool) {
    let parsed = grouped_model(class, grouping);
    let pop = prepared(&parsed, vec![cell_subject("1", &parsed, rows)]);
    let advice = dw_advice(&parsed.model, &parsed.fit_options, &pop);
    let msg = dw_autocorrelation_warning(1.20, &advice).expect("below 1.5");
    (advice, msg, all_in_time(&pop))
}

/// A closed-form PK model, under every grouping and layout: today's text, byte for
/// byte.
///
/// Mutation that must redden this: `is_algebraic()` → `ode_spec.is_none()` in
/// `dw_advice` (true for a closed form too); any rewording of today's text.
#[test]
fn pk_analytic_keeps_todays_text_under_every_grouping() {
    for g in GROUPINGS {
        let parsed = grouped_model(Class::PkAnalytic, g);
        assert!(
            parsed.model.ode_spec.is_none(),
            "the row exists to straddle `ode_spec.is_none()`; {g:?}"
        );
        for rows in LAYOUTS {
            let (advice, msg, _) = cell(Class::PkAnalytic, g, rows);
            assert_eq!(advice, DwAdvice::Compartmental, "{g:?} {rows:?}");
            assert_eq!(msg, COMPARTMENTAL, "{g:?} {rows:?}");
            assert!(
                !msg.contains("`correlation") && !msg.contains("kappa"),
                "{g:?} {rows:?}: {msg}"
            );
        }
    }
}

/// An ODE model, under every grouping and layout: today's text, byte for byte — the
/// #1285 property, no SDE / `[diffusion]` sentence, and no extra one either.
#[test]
fn ode_keeps_todays_text_under_every_grouping() {
    for g in GROUPINGS {
        for rows in LAYOUTS {
            let (advice, msg, _) = cell(Class::Ode, g, rows);
            assert_eq!(advice, DwAdvice::Compartmental, "{g:?} {rows:?}");
            assert_eq!(msg, COMPARTMENTAL, "{g:?} {rows:?}");
            for banned in ["`correlation", "SDE", "diffusion"] {
                assert!(!msg.contains(banned), "{g:?} {rows:?}: `{banned}` in {msg}");
            }
        }
    }
}

/// The PK remedies a compartment-free model cannot take, and `cs`, which over a
/// kappa-carrying grouping is refused.
const NOT_FOR_ALGEBRAIC: [&str; 4] = ["transit", "compartment", "ka/F", "cs("];

/// Compartment-free, no grouping: declare one and correlate by it, with a
/// placeholder column, never a concrete one. Whatever the clock does.
#[test]
fn algebraic_without_grouping_names_a_level_and_a_placeholder_column() {
    for rows in LAYOUTS {
        let (advice, msg, in_time) = cell(Class::Algebraic, Grouping::None, rows);
        assert!(in_time, "{rows:?}: no occasions, nothing to split at");
        assert_eq!(advice, DwAdvice::Algebraic { group: None }, "{rows:?}");
        assert_eq!(msg, ALGEBRAIC_UNGROUPED, "{rows:?}");
        for must in [
            "kappa … at",
            "correlation ar1(RHO) by COL",
            "innermost level's name",
            "foce",
            "focei",
            "car1",
        ] {
            assert!(msg.contains(must), "missing `{must}`: {msg}");
        }
        for banned in NOT_FOR_ALGEBRAIC.iter().chain(&["OCC", "ARM", "STUDY"]) {
            assert!(!msg.contains(banned), "`{banned}` in {msg}");
        }
    }
}

/// A2. Compartment-free with `iov_column = OCC` on **parallel** occasions: the
/// statistic split at the column, so correlate by it.
///
/// Mutation that must redden this: `dw_advice` never naming a group.
#[test]
fn algebraic_with_a_parallel_iov_column_correlates_by_the_column() {
    let (advice, msg, in_time) = cell(Class::Algebraic, Grouping::IovColumn, Rows::Parallel);
    assert!(!in_time, "the fixture must be parallel");
    assert_eq!(
        advice,
        DwAdvice::Algebraic {
            group: Some("OCC".into())
        }
    );
    assert_eq!(msg, algebraic_grouped("OCC"));
    for must in [
        "by OCC`",
        "`OCC` group",
        "ar1(RHO)",
        "car1(RHO)",
        "foce or focei",
    ] {
        assert!(msg.contains(must), "missing `{must}`: {msg}");
    }
    for banned in NOT_FOR_ALGEBRAIC.iter().chain(&["COL", "kappa"]) {
        assert!(!msg.contains(banned), "`{banned}` in {msg}");
    }
}

/// A1 (#2829). Compartment-free with `iov_column = OCC` on occasions that **run in
/// time**: the statistic was taken over the whole subject, so the advice names no
/// column. The twin of the test above: the same model and labels, only the clock
/// differs.
///
/// Mutation that must redden this, and only this among the one-subject cells: the
/// `occasions_run_in_time` read dropped from `dw_advice` (`by OCC` again).
#[test]
fn algebraic_with_an_iov_column_in_time_gets_the_ungrouped_text() {
    let parsed = grouped_model(Class::Algebraic, Grouping::IovColumn);
    assert_eq!(parsed.fit_options.iov_column.as_deref(), Some("OCC"));
    assert!(
        !crate::api::run::occasions_are_derived(&parsed.model, &parsed.fit_options),
        "the straddle: a column wrote these labels, not a rule"
    );
    let (advice, msg, in_time) = cell(Class::Algebraic, Grouping::IovColumn, Rows::InTime);
    assert!(in_time, "the fixture must run in time");
    assert_eq!(advice, DwAdvice::Algebraic { group: None });
    assert_eq!(msg, ALGEBRAIC_UNGROUPED);
    assert!(
        msg.contains("by COL` line") && msg.contains("COL any"),
        "{msg}"
    );
    assert!(!msg.contains("OCC"), "{msg}");
}

/// A4 (#2829, plan amendment 1). A population of one subject in time and one
/// parallel, under `iov_column = OCC`: the statistic split **some** subject at the
/// column, so the advice names it.
///
/// Mutation that must redden this: "some subject" → "every subject" in `dw_advice`.
#[test]
fn algebraic_iov_column_split_in_some_subjects_correlates_by_the_column() {
    let parsed = grouped_model(Class::Algebraic, Grouping::IovColumn);
    for order in [
        [Rows::InTime, Rows::Parallel],
        [Rows::Parallel, Rows::InTime],
    ] {
        let pop = prepared(
            &parsed,
            vec![
                cell_subject("1", &parsed, order[0]),
                cell_subject("2", &parsed, order[1]),
            ],
        );
        let in_time: Vec<bool> = pop
            .subjects
            .iter()
            .map(crate::stats::residual_error::occasions_run_in_time)
            .collect();
        assert!(
            in_time.contains(&true) && in_time.contains(&false),
            "the population must be mixed: {in_time:?}"
        );
        let advice = dw_advice(&parsed.model, &parsed.fit_options, &pop);
        let msg = dw_autocorrelation_warning(1.20, &advice).expect("below 1.5");
        assert_eq!(msg, algebraic_grouped("OCC"), "{order:?}");
        assert!(
            msg.contains("`OCC` group") && msg.contains("by OCC`"),
            "{order:?}: {msg}"
        );
        assert!(!msg.contains("COL any"), "{order:?}: {msg}");
    }
}

/// Compartment-free with one `level ARM by ARMID in ID`: the level's **column** —
/// a one-level model has no named level a `by` can read — where its groups are
/// parallel, and the ungrouped text where they run in time (a single level is
/// `iov_column` by construction, so it reads the clock as the column does, #2829).
///
/// Mutation that must redden this: the level's name in place of its column.
#[test]
fn algebraic_single_level_correlates_by_its_column() {
    let (advice, msg, in_time) = cell(Class::Algebraic, Grouping::SingleLevel, Rows::Parallel);
    assert!(!in_time);
    assert_eq!(
        advice,
        DwAdvice::Algebraic {
            group: Some("ARMID".into())
        }
    );
    assert_eq!(msg, algebraic_grouped("ARMID"));
    assert!(msg.contains("by ARMID`"), "{msg}");
    assert!(!msg.contains("by ARM`") && !msg.contains("`ARM`"), "{msg}");
    for banned in NOT_FOR_ALGEBRAIC {
        assert!(!msg.contains(banned), "`{banned}` in {msg}");
    }
    let (advice, msg, in_time) = cell(Class::Algebraic, Grouping::SingleLevel, Rows::InTime);
    assert!(in_time);
    assert_eq!(advice, DwAdvice::Algebraic { group: None });
    assert!(msg == ALGEBRAIC_UNGROUPED && !msg.contains("ARM"), "{msg}");
}

/// Compartment-free with nested `STUDY > ARM`: the **innermost** level's **name**,
/// which reads its groups nested by key, where the cells are parallel. Cells that
/// run in time are one series like any other occasions (no exemption for a declared
/// level, #2829), and get the ungrouped text.
///
/// Mutations that must redden this: the column in place of the name (the opposite
/// direction of the single-level row); the outermost level in place of the
/// innermost; the nested arm placed before the `occasions_run_in_time` read.
#[test]
fn algebraic_nested_levels_correlate_by_the_innermost_level_name() {
    let (advice, msg, in_time) = cell(Class::Algebraic, Grouping::Nested, Rows::Parallel);
    assert!(!in_time);
    assert_eq!(
        advice,
        DwAdvice::Algebraic {
            group: Some("ARM".into())
        }
    );
    assert_eq!(msg, algebraic_grouped("ARM"));
    assert!(msg.contains("by ARM`"), "{msg}");
    for banned in NOT_FOR_ALGEBRAIC.iter().chain(&["ARMID", "STUDY"]) {
        assert!(!msg.contains(banned), "`{banned}` in {msg}");
    }
    let (advice, msg, in_time) = cell(Class::Algebraic, Grouping::Nested, Rows::InTime);
    assert!(in_time);
    assert_eq!(advice, DwAdvice::Algebraic { group: None });
    assert!(msg == ALGEBRAIC_UNGROUPED && !msg.contains("ARM"), "{msg}");
}

/// Compartment-free with `iov_occasion = time(...)` or `dose` and a kappa, on one
/// running clock: the occasions come from the timeline through
/// `fitted_population`, they run in time (asserted), and the statistic is taken over
/// the whole subject. The ungrouped text, although the fixture also sets
/// `iov_column`.
#[test]
fn algebraic_derived_occasions_get_the_ungrouped_text() {
    for g in [Grouping::IovTime, Grouping::IovDose] {
        let parsed = grouped_model(Class::Algebraic, g);
        assert_eq!(
            parsed.fit_options.iov_column.as_deref(),
            Some("OCC"),
            "{g:?}: a column is set, and must not reach the advice"
        );
        assert!(
            crate::api::run::occasions_are_derived(&parsed.model, &parsed.fit_options),
            "{g:?}: the fixture must derive its occasions"
        );
        let pop = prepared(&parsed, vec![cell_subject("1", &parsed, Rows::InTime)]);
        assert!(
            pop.subjects[0].occasions.windows(2).any(|w| w[0] != w[1]),
            "{g:?}: the rule must derive more than one occasion: {:?}",
            pop.subjects[0].occasions
        );
        let (advice, msg, in_time) = cell(Class::Algebraic, g, Rows::InTime);
        assert!(in_time, "{g:?}: derived occasions on a running clock");
        assert_eq!(advice, DwAdvice::Algebraic { group: None }, "{g:?}");
        assert_eq!(msg, ALGEBRAIC_UNGROUPED, "{g:?}");
        assert!(!msg.contains("OCC"), "{g:?}: {msg}");
    }
}

/// A derived rule on a clock that **restarts** (`time(24, 48)` over rows at
/// `6, 30, 6, 30`): the window labels come back, `[0, 1, 0, 1]`, so the statistic is
/// split at them — but no column labels a window, and the rule overrides the
/// `iov_column` the fixture also sets. The ungrouped text, not `by OCC`.
///
/// This is the one cell where the derived-rule arm of `dw_advice` and the
/// `occasions_run_in_time` read disagree, so it is the arm's own test.
///
/// Mutation that must redden this, and only this: the `occasions_are_derived` arm
/// dropped from `dw_advice`.
#[test]
fn algebraic_derived_windows_on_a_restarting_clock_name_no_column() {
    let parsed = grouped_model(Class::Algebraic, Grouping::IovTime);
    assert_eq!(parsed.fit_options.iov_column.as_deref(), Some("OCC"));
    let pop = prepared(&parsed, vec![cell_subject("1", &parsed, Rows::Parallel)]);
    assert_eq!(pop.subjects[0].occasions, vec![0, 1, 0, 1]);
    let (advice, msg, in_time) = cell(Class::Algebraic, Grouping::IovTime, Rows::Parallel);
    assert!(
        !in_time,
        "the straddle: derived labels that do not run in time"
    );
    assert_eq!(advice, DwAdvice::Algebraic { group: None });
    assert_eq!(msg, ALGEBRAIC_UNGROUPED);
    assert!(msg.contains("COL any") && !msg.contains("OCC"), "{msg}");
}

/// A3. Compartment-free with `iov_column = OCC` and `iov_occasion = time(...)` but
/// no kappa: the fit derives no occasions, so the column's labels stand (#2815
/// review round 1, row 1). Parallel, the advice correlates by the column; in time,
/// it names none (#2829).
///
/// Mutations that must redden this: dropping the `n_kappa > 0` term of
/// `occasions_are_derived` (the parallel half loses its column); dropping the
/// `occasions_run_in_time` read (the in-time half gains one).
#[test]
fn algebraic_derived_rule_without_a_kappa_keeps_the_column() {
    let parsed = grouped_model(Class::Algebraic, Grouping::IovTimeNoKappa);
    assert_ne!(
        parsed.fit_options.iov_occasion,
        crate::types::IovOccasionRule::Column,
        "the straddle: a derived rule is set, and only the missing kappa voids it"
    );
    assert!(!crate::api::run::occasions_are_derived(
        &parsed.model,
        &parsed.fit_options
    ));
    let pop = prepared(&parsed, vec![cell_subject("1", &parsed, Rows::Parallel)]);
    assert_eq!(
        pop.subjects[0].occasions,
        vec![1, 1, 2, 2],
        "the column's labels, not the windows'"
    );
    let (advice, msg, in_time) = cell(Class::Algebraic, Grouping::IovTimeNoKappa, Rows::Parallel);
    assert!(!in_time);
    assert_eq!(
        advice,
        DwAdvice::Algebraic {
            group: Some("OCC".into())
        }
    );
    assert_eq!(msg, algebraic_grouped("OCC"));
    assert!(msg.contains("by OCC`") && !msg.contains("COL"), "{msg}");
    let (advice, msg, in_time) = cell(Class::Algebraic, Grouping::IovTimeNoKappa, Rows::InTime);
    assert!(in_time);
    assert_eq!(advice, DwAdvice::Algebraic { group: None });
    assert!(msg.contains("COL any") && !msg.contains("OCC"), "{msg}");
}

/// The negative branch is the same for every model: no model-class remedy.
#[test]
fn negative_text_is_the_same_for_every_advice() {
    let negative = "Negative IWRES autocorrelation detected (Durbin-Watson = 2.80). \
                    Possible over-parameterization or misspecified error model.";
    for advice in [
        DwAdvice::Compartmental,
        DwAdvice::Algebraic { group: None },
        DwAdvice::Algebraic {
            group: Some("ARM".into()),
        },
    ] {
        let msg = dw_autocorrelation_warning(2.80, &advice).expect("above 2.5");
        assert_eq!(msg, negative, "{advice:?}");
        for banned in ["`correlation", "transit", "kappa"] {
            assert!(!msg.contains(banned), "{advice:?}: `{banned}`");
        }
        for dw in [1.5, 2.0, 2.5, f64::NAN] {
            assert_eq!(
                dw_autocorrelation_warning(dw, &advice),
                None,
                "{advice:?} {dw}"
            );
        }
    }
}
