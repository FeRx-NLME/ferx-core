//! `profile`d θ level blocks (#1065): the per-subject active sets, the route and
//! scope refusals, the parameter count and the output fields.

use super::*;
use crate::parser::model_parser::parse_full_model;
use crate::types::{CompiledModel, Population, Subject};

/// An algebraic MBMA model: `BASE = TVE0 + <read> + ETA_E0`, additive error. `decl`
/// is the block's declaration line.
fn mbma(decl: &str, read: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVE0(45.0, 0.0, 200.0)
  {decl}
  omega ETA_E0 ~ 4.0
  sigma ADD_ERR ~ 1.0 FIX

[covariates]
  STUDY categorical

[individual_parameters]
  BASE = TVE0 + {read} + ETA_E0

[structural_model]
  y = BASE

[error_model]
  DV ~ additive(ADD_ERR)
"#
    )
}

/// Two arms (subjects) per study, `n_studies` studies, each arm observed at
/// `TIME = 1..=n_times`; `STUDY` a subject-level covariate. Two subjects share each
/// study so a study-level block and a per-subject η are told apart. No doses: the
/// model is compartment-free.
fn population(n_studies: usize, n_times: usize) -> Population {
    let subjects = (0..2 * n_studies)
        .map(|i| {
            let mut covariates = HashMap::new();
            covariates.insert("STUDY".to_string(), (i / 2 + 1) as f64);
            Subject {
                id: format!("{}", i + 1),
                obs_times: (1..=n_times).map(|t| t as f64).collect(),
                observations: vec![45.0; n_times],
                obs_cmts: vec![1; n_times],
                cens: vec![0; n_times],
                covariates,
                ..Subject::default()
            }
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec!["STUDY".to_string()],
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// Bind `text` against `pop` (writing the level columns), returning the model.
fn bind(text: &str, pop: &mut Population) -> CompiledModel {
    let mut parsed = parse_full_model(text).unwrap_or_else(|e| panic!("{e}"));
    crate::api::bind_theta_levels(&mut parsed, text, pop).unwrap_or_else(|e| panic!("{e}"));
    parsed.model
}

// ── ActiveLevels ────────────────────────────────────────────────────────────

/// A block keyed on a subject-constant column: each subject reads one level, its
/// own study's, from the baseline map (no per-row snapshots are written).
#[test]
fn a_subject_constant_index_gives_each_subject_its_one_level() {
    let mut pop = population(3, 4);
    let model = bind(
        &mbma("theta PLACEBO[STUDY](0.0) profile", "PLACEBO"),
        &mut pop,
    );
    assert!(
        pop.subjects.iter().all(|s| s.obs_covariates.is_empty()),
        "the fixture must stay on the baseline-only path"
    );
    let active = active_levels(&model, &pop);
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].block, "PLACEBO");
    assert_eq!(
        active[0].per_subject,
        [vec![0], vec![0], vec![1], vec![1], vec![2], vec![2]]
    );
}

/// A block keyed on `[STUDY, TIME]`: the index moves row by row, so each subject
/// reads its own four cells. Reading the baseline alone sees one cell per subject.
#[test]
fn a_per_row_index_gives_each_subject_every_cell_it_observes() {
    let mut pop = population(3, 4);
    let model = bind(
        &mbma("theta PLACEBO[STUDY, TIME](0.0) profile", "PLACEBO"),
        &mut pop,
    );
    assert!(
        pop.subjects.iter().all(|s| !s.obs_covariates.is_empty()),
        "the fixture must take the per-row path"
    );
    let active = active_levels(&model, &pop);
    let study = |s: usize| (4 * s..4 * s + 4).collect::<Vec<usize>>();
    assert_eq!(
        active[0].per_subject,
        [study(0), study(0), study(1), study(1), study(2), study(2)]
    );
}

/// The counted form, read through the user's own plain column, gives the same sets
/// as the column form; an unprofiled block gets no entry at all.
#[test]
fn the_counted_form_reads_its_plain_index_column_and_unprofiled_blocks_are_skipped() {
    let mut pop = population(3, 2);
    let model = bind(
        &mbma("theta PLACEBO[3](0.0) profile", "PLACEBO[STUDY]"),
        &mut pop,
    );
    let active = active_levels(&model, &pop);
    assert_eq!(
        active[0].per_subject,
        [vec![0], vec![0], vec![1], vec![1], vec![2], vec![2]]
    );

    let mut pop = population(3, 2);
    let plain = bind(&mbma("theta PLACEBO[3](0.0)", "PLACEBO[STUDY]"), &mut pop);
    assert!(active_levels(&plain, &pop).is_empty());
}

// ── E_PROFILE_SCOPE: what a profiled block can be fitted with ───────────────

use crate::types::{CovarianceMethod, EstimationMethod, FitOptions};

const FRAME: &str = "theta PLACEBO[...] is `profile`d, ";
const REMOVE: &str = "Remove `profile` to estimate the levels as ordinary θ.";

fn counted() -> String {
    mbma("theta PLACEBO[3](0.0) profile", "PLACEBO[STUDY]")
}

fn model_of(text: &str) -> CompiledModel {
    parse_full_model(text)
        .unwrap_or_else(|e| panic!("{e}"))
        .model
}

/// The `E_PROFILE_SCOPE` messages `check_model_options` raises, in order.
fn scope(model: &CompiledModel, options: &FitOptions) -> Vec<String> {
    crate::api::check_model_options(model, options)
        .into_iter()
        .filter(|d| d.code == "E_PROFILE_SCOPE")
        .map(|d| {
            assert_eq!(d.block.as_deref(), Some("parameters"), "{}", d.message);
            d.message
        })
        .collect()
}

/// Exactly one refusal, framed, with `must` and its fix sentence; returns it for the
/// must-not-say checks.
fn one_refusal(msgs: &[String], must: &str, fix: &str) -> String {
    assert_eq!(msgs.len(), 1, "{msgs:#?}");
    let m = &msgs[0];
    for s in [FRAME, must, fix] {
        assert!(m.contains(s), "missing {s:?} in {m}");
    }
    m.clone()
}

/// The control: a profiled block in scope is not refused (#1065 PR2 lifted PR1's
/// `not implemented` refusal), and neither is an unprofiled model.
#[test]
fn an_in_scope_profiled_block_is_not_refused() {
    let msgs = scope(&model_of(&counted()), &FitOptions::default());
    assert!(msgs.is_empty(), "{msgs:#?}");
    let plain = model_of(&mbma("theta PLACEBO[3](0.0)", "PLACEBO[STUDY]"));
    assert!(scope(&plain, &FitOptions::default()).is_empty());
}

/// Every estimator outside `foce`/`focei`, alone or as a chain step, and quadrature.
/// The message names what it saw and never claims a fallback.
#[test]
fn profile_refuses_every_method_outside_foce_and_focei() {
    for m in [
        EstimationMethod::Saem,
        EstimationMethod::Imp,
        EstimationMethod::Laplace,
        EstimationMethod::FoceGn,
        EstimationMethod::Bayes,
    ] {
        let alone = FitOptions {
            method: m,
            ..FitOptions::default()
        };
        let chain = FitOptions {
            methods: vec![EstimationMethod::FoceI, m],
            ..FitOptions::default()
        };
        for opts in [alone, chain] {
            let msg = one_refusal(
                &scope(&model_of(&counted()), &opts),
                &format!(
                    "which only `foce` and `focei` estimate; this fit runs `{}`.",
                    m.label()
                ),
                REMOVE,
            );
            assert!(!msg.contains("fall"), "{msg}");
        }
    }
    for ok in [EstimationMethod::Foce, EstimationMethod::FoceI] {
        let opts = FitOptions {
            method: ok,
            ..FitOptions::default()
        };
        assert!(scope(&model_of(&counted()), &opts).is_empty(), "{ok:?}");
    }
    let agq = FitOptions {
        method: EstimationMethod::FoceI,
        n_agq: 3,
        ..FitOptions::default()
    };
    let msg = one_refusal(
        &scope(&model_of(&counted()), &agq),
        "which only `foce` and `focei` estimate, at `n_agq = 1`; this fit asks for \
         `n_agq = 3` quadrature.",
        REMOVE,
    );
    assert!(!msg.contains("fall"), "{msg}");
}

/// Proportional, power, combined and a log-transformed additive error, by name.
#[test]
fn profile_refuses_a_residual_error_that_is_not_additive() {
    for (error, extra_sigma, must) in [
        (
            "DV ~ proportional(ADD_ERR)",
            "",
            "which needs an additive residual error; this model's is proportional (or power), \
             whose variance reads the prediction.",
        ),
        (
            "DV ~ power(ADD_ERR, 0.5)",
            "",
            "which needs an additive residual error; this model's is proportional (or power), \
             whose variance reads the prediction.",
        ),
        (
            "DV ~ combined(PROP_ERR, ADD_ERR)",
            "  sigma PROP_ERR ~ 0.01\n",
            "which needs an additive residual error; this model's is combined, whose variance \
             reads the prediction.",
        ),
        (
            "log(DV) ~ additive(ADD_ERR)",
            "",
            "which needs an additive residual error on the data's own scale; this model \
             transforms both sides by a log, so the levels enter through `log(...)`.",
        ),
    ] {
        let text = counted()
            .replace("DV ~ additive(ADD_ERR)", error)
            .replace("  sigma ADD_ERR", &format!("{extra_sigma}  sigma ADD_ERR"));
        one_refusal(
            &scope(&model_of(&text), &FitOptions::default()),
            must,
            REMOVE,
        );
    }
}

/// M3 censoring: armed through the options, as `fit()` arms it.
#[test]
fn profile_refuses_m3_censoring() {
    let opts = FitOptions {
        bloq_method: Some(crate::types::BloqMethod::M3),
        ..FitOptions::default()
    };
    one_refusal(
        &scope(&model_of(&counted()), &opts),
        "which needs a Gaussian likelihood on every row; this fit scores censored rows by M3, \
         a normal-tail probability.",
        REMOVE,
    );
}

/// A `correlation` line: not yet, with the issue, never "invalid".
#[test]
fn profile_refuses_a_correlation_line_as_not_yet_supported() {
    let text = counted()
        .replace(
            "  omega ETA_E0",
            "  theta RHO(0.3, -0.2, 0.95)\n  omega ETA_E0",
        )
        .replace(
            "  STUDY categorical",
            "  STUDY categorical\n  ARM categorical",
        )
        .replace(
            "DV ~ additive(ADD_ERR)",
            "DV ~ additive(ADD_ERR)\n  correlation cs(RHO) by ARM",
        );
    let msg = one_refusal(
        &scope(&model_of(&text), &FitOptions::default()),
        "and a `correlation` line is not yet supported together with `profile` (#1082).",
        "Remove one of the two for now.",
    );
    assert!(!msg.to_lowercase().contains("invalid"), "{msg}");
}

/// `[mixture]` parsed; `[diffusion]` and FREM set on the parsed model by hand (the
/// check reads `is_sde()` and `frem_config`, and those blocks need an ODE model or a
/// FREM dataset this algebraic fixture does not have).
#[test]
fn profile_refuses_mixture_diffusion_and_frem() {
    let mix = counted()
        .replace(
            "  omega ETA_E0",
            "  theta MIXL(0.0, -5.0, 5.0)\n  omega ETA_E0",
        )
        .replace(
            "[error_model]",
            "[mixture]\n  nsub = 2\n  logit(1) = MIXL\n\n[error_model]",
        );
    one_refusal(
        &scope(&model_of(&mix), &FitOptions::default()),
        "which does not yet combine with a `[mixture]`.",
        REMOVE,
    );

    let mut sde = model_of(&counted());
    sde.diffusion_theta_start = Some(sde.n_theta);
    one_refusal(
        &scope(&sde, &FitOptions::default()),
        "which does not yet combine with a `[diffusion]` block.",
        REMOVE,
    );

    let mut frem = model_of(&counted());
    frem.frem_config = Some(crate::types::FremConfig {
        fremtype_to_indices: HashMap::new(),
        covariate_sigma_index: 0,
    });
    one_refusal(
        &scope(&frem, &FitOptions::default()),
        "which does not yet combine with a FREM covariate model.",
        REMOVE,
    );
}

/// An explicit `s` or `rsr` covariance; a defaulted one, or no covariance step, is
/// not refused.
#[test]
fn profile_refuses_an_s_or_sandwich_covariance() {
    for m in [CovarianceMethod::CrossProduct, CovarianceMethod::Sandwich] {
        let opts = FitOptions {
            run_covariance_step: true,
            covariance_method: m,
            covariance_method_set: true,
            ..FitOptions::default()
        };
        one_refusal(
            &scope(&model_of(&counted()), &opts),
            &format!(
                "whose covariance step computes `r` only for now; this fit asks for \
                 `covariance_method = {}`.",
                m.label()
            ),
            "Use `covariance_method = r`, or remove `profile`.",
        );
        let off = FitOptions {
            run_covariance_step: false,
            covariance_method: m,
            covariance_method_set: true,
            ..FitOptions::default()
        };
        assert!(scope(&model_of(&counted()), &off).is_empty(), "no step");
    }
}

/// Two profiled blocks are both named, in declaration order.
#[test]
fn every_profiled_block_is_named() {
    let text = counted()
        .replace(
            "  omega ETA_E0",
            "  theta ARMFX[2](0.0) profile\n  omega ETA_E0",
        )
        .replace(
            "  STUDY categorical",
            "  STUDY categorical\n  ARM categorical",
        )
        // Each block read by its own prediction: one may read only one profiled block,
        // and since #2103 review #8 a profiled block must be read by some prediction.
        .replace(
            "  y = BASE",
            "  y[CMT=1] = BASE\n  y[CMT=2] = TVE0 + ARMFX[ARM]",
        );
    let model = model_of(&text);
    let saem = FitOptions {
        method: EstimationMethod::Saem,
        ..FitOptions::default()
    };
    let msgs = scope(&model, &saem);
    assert!(
        msgs[0].starts_with("theta PLACEBO[...] and ARMFX[...] are `profile`d, "),
        "{}",
        msgs[0]
    );
}

/// A binary endpoint beside the Gaussian rows.
#[cfg(feature = "survival")]
#[test]
fn profile_refuses_a_model_with_a_non_gaussian_endpoint() {
    let text = counted()
        .replace(
            "  omega ETA_E0",
            "  theta TH0(0.0, -10.0, 10.0)\n  omega ETA_E0",
        )
        .replace(
            "[error_model]",
            "[binary_model]\n  cmt   = 3\n  logit = TH0\n\n[error_model]",
        );
    one_refusal(
        &scope(&model_of(&text), &FitOptions::default()),
        "which needs a Gaussian likelihood on every row; this model has a binary endpoint.",
        REMOVE,
    );
}

/// The parse-time codes reach the check report as their own code, on `[parameters]`.
#[test]
fn the_parse_time_profile_codes_map_to_their_diagnostics() {
    for (text, code) in [
        (
            mbma("theta PLACEBO(0.0) profile", "PLACEBO"),
            "E_PROFILE_SCOPE",
        ),
        (
            mbma("theta PLACEBO[3](0.0) profile", "exp(PLACEBO[STUDY])"),
            "E_PROFILE_NOT_LINEAR",
        ),
    ] {
        let err = parse_full_model(&text).err().expect("refused");
        let d = crate::api::validation::parse_error_to_diagnostic(&err);
        assert_eq!(d.code, code, "{err}");
        assert_eq!(d.block.as_deref(), Some("parameters"));
        assert!(!d.message.starts_with(code), "{}", d.message);
    }
}

// ── P1: a profiled block's levels count as estimated parameters ─────────────

/// The free count and BIC tally of `model` under `held` (a packed held mask), via
/// the production counting hook.
fn count_of(model: &CompiledModel, held: &[bool]) -> (usize, crate::types::BicInputs) {
    let mask = crate::model_selection::counted_held_mask(model, &model.default_params, held);
    let n = mask.iter().filter(|&&h| !h).count();
    let bic = crate::model_selection::bic_inputs_for(model, &model.default_params, &mask, 100);
    (n, bic)
}

/// PR2's shape: the optimizer holds every profiled level, the way it holds a FIX θ.
fn held_like_the_profile_solve(model: &CompiledModel) -> Vec<bool> {
    let mut held = crate::estimation::parameterization::packed_fixed_mask(&model.default_params);
    let idx = model
        .indiv_param_partials
        .theta_blocks
        .profiled_theta_indices();
    assert!(!idx.is_empty(), "the fixture must have profiled levels");
    for i in idx {
        held[i] = true;
    }
    held
}

/// P1. With every profiled level held from the optimizer, `n_parameters` and the
/// BIC tally equal the unprofiled model's — the counted form, and the column form
/// under `sum_to_zero_within`, whose dependent levels are no θ. A level ferx froze
/// stays uncounted on both sides.
#[test]
fn a_profiled_fit_counts_the_parameters_the_unprofiled_fit_counts() {
    let pairs = [
        (
            "theta PLACEBO[3](0.0) profile",
            "theta PLACEBO[3](0.0, -10.0, 10.0)",
            "PLACEBO[STUDY]",
        ),
        (
            "theta PLACEBO[STUDY, TIME, contrast = sum_to_zero_within](0.0) profile",
            "theta PLACEBO[STUDY, TIME, contrast = sum_to_zero_within](0.0, -10.0, 10.0)",
            "PLACEBO",
        ),
    ];
    for (p_decl, u_decl, read) in pairs {
        let p = bind(&mbma(p_decl, read), &mut population(3, 4));
        let u = bind(&mbma(u_decl, read), &mut population(3, 4));
        let u_held = crate::estimation::parameterization::packed_fixed_mask(&u.default_params);
        let want = count_of(&u, &u_held);
        assert_eq!(
            count_of(&p, &held_like_the_profile_solve(&p)),
            want,
            "{p_decl}"
        );
        // The straddle: held and not counted back, the levels would vanish.
        let searched = held_like_the_profile_solve(&p)
            .iter()
            .filter(|&&h| !h)
            .count();
        assert!(searched < want.0, "{p_decl}: {searched} vs {}", want.0);

        // One level frozen (as #1891's `theta_fixed` freezes one) on both sides.
        let first = p.indiv_param_partials.theta_blocks.profiled_theta_indices()[0];
        let (mut p, mut u) = (p, u);
        p.default_params.theta_fixed[first] = true;
        u.default_params.theta_fixed[first] = true;
        let u_held = crate::estimation::parameterization::packed_fixed_mask(&u.default_params);
        let want_frozen = count_of(&u, &u_held);
        assert_eq!(want_frozen.0, want.0 - 1);
        assert_eq!(
            count_of(&p, &held_like_the_profile_solve(&p)),
            want_frozen,
            "{p_decl}: frozen"
        );
    }
}

// ── #2103 review round 1 ────────────────────────────────────────────────────

/// Review #2: one read through a computed index (`PLACEBO[4 - STUDY]`) beside a
/// plain one gives every subject every level — the rows read the reversed level,
/// so the plain column alone would name the wrong one. The control, both reads
/// plain, keeps the exact sets.
#[test]
fn a_computed_index_read_gives_every_subject_every_level() {
    let with_d = |read: &str| {
        mbma("theta PLACEBO[3](0.0) profile", read).replace(
            "[individual_parameters]\n",
            "[individual_parameters]\n  D = PLACEBO[STUDY]\n",
        )
    };
    let mut pop = population(3, 2);
    let model = bind(&with_d("PLACEBO[4 - STUDY]"), &mut pop);
    assert_eq!(
        active_levels(&model, &pop)[0].per_subject,
        vec![vec![0, 1, 2]; 6]
    );

    let mut pop = population(3, 2);
    let model = bind(&with_d("PLACEBO[STUDY]"), &mut pop);
    assert_eq!(
        active_levels(&model, &pop)[0].per_subject,
        [vec![0], vec![0], vec![1], vec![1], vec![2], vec![2]]
    );
}

/// Review #3b: a defaulted covariance method is not refused, whatever its value —
/// the gate reads an explicit choice. (Scale routing of a defaulted `r` to `s` is
/// PR2's: the dimension it routes on is the one PR2 holds.)
#[test]
fn a_defaulted_covariance_method_is_not_refused() {
    let defaulted = FitOptions {
        run_covariance_step: true,
        covariance_method: CovarianceMethod::CrossProduct,
        covariance_method_set: false,
        ..FitOptions::default()
    };
    let msgs = scope(&model_of(&counted()), &defaulted);
    assert!(msgs.is_empty(), "{msgs:#?}");
}

// ── PR2: what the profiled outer loop does not run ──────────────────────────

/// The profiled outer loop has no trust-region or MMA arm: each is refused, naming the
/// optimizer it saw and the four it runs. The L-BFGS spellings and BOBYQA/SLSQP are
/// not.
#[test]
fn profile_refuses_an_optimizer_its_outer_loop_does_not_run() {
    use crate::types::Optimizer;
    for opt in [Optimizer::TrustRegion, Optimizer::Mma] {
        let o = FitOptions {
            optimizer: opt,
            ..FitOptions::default()
        };
        one_refusal(
            &scope(&model_of(&counted()), &o),
            &format!(
                "whose outer loop runs `auto`, `bobyqa`, `slsqp` or an L-BFGS; this fit asks \
                 for `optimizer = {}`.",
                opt.label()
            ),
            "Choose one of those, or remove `profile`.",
        );
    }
    for opt in [
        Optimizer::Auto,
        Optimizer::Bobyqa,
        Optimizer::Slsqp,
        Optimizer::NloptLbfgs,
        Optimizer::Lbfgs,
        Optimizer::Bfgs,
    ] {
        let o = FitOptions {
            optimizer: opt,
            ..FitOptions::default()
        };
        assert!(scope(&model_of(&counted()), &o).is_empty(), "{opt:?}");
    }
}

/// `global_search` and a parameter prior: neither is applied by the profiled outer
/// loop, so each is refused rather than silently dropped.
#[test]
fn profile_refuses_a_global_search_and_a_prior() {
    let o = FitOptions {
        global_search: true,
        ..FitOptions::default()
    };
    one_refusal(
        &scope(&model_of(&counted()), &o),
        "which does not yet run a `global_search` pre-search.",
        "Set `global_search = false`, or remove `profile`.",
    );
    let priored = counted().replace(
        "theta TVE0(45.0, 0.0, 200.0)",
        "theta TVE0(45.0, 0.0, 200.0) prior(45.0, rse = 25%)",
    );
    one_refusal(
        &scope(&model_of(&priored), &FitOptions::default()),
        "which does not yet combine with a parameter prior.",
        "Remove the prior, or remove `profile`.",
    );
}

/// Review #6: the time-to-event arm, named, and no "also".
#[cfg(feature = "survival")]
#[test]
fn profile_refuses_a_model_with_a_time_to_event_endpoint() {
    let text = counted()
        .replace("  omega ETA_E0", "  theta TVLAMBDA(0.05, 0.001, 10.0)\n  omega ETA_E0")
        .replace(
            "[error_model]",
            "[event_model]\n  cmt    = 2\n  family = exponential\n  scale  = TVLAMBDA\n\n[error_model]",
        );
    let msg = one_refusal(
        &scope(&model_of(&text), &FitOptions::default()),
        "which needs a Gaussian likelihood on every row; this model has a time-to-event endpoint.",
        REMOVE,
    );
    assert!(!msg.contains("also"), "{msg}");
}

/// Review #6: the third arm, which only a CTMM endpoint reaches.
#[cfg(feature = "markov")]
#[test]
fn profile_refuses_a_model_with_a_markov_endpoint() {
    let text = counted()
        .replace("  omega ETA_E0", "  theta LQ01(-0.7, -6.0, 3.0)\n  theta LQ10(-1.2, -6.0, 3.0)\n  omega ETA_E0")
        .replace(
            "[error_model]",
            "[markov_model]\n  type   = ctmm\n  cmt    = 5\n  states = [awake=0, asleep=1]\n  transition awake  -> asleep = exp(LQ01)\n  transition asleep -> awake  = exp(LQ10)\n\n[error_model]",
        );
    one_refusal(
        &scope(&model_of(&text), &FitOptions::default()),
        "which needs a Gaussian likelihood on every row; this model has a Markov (CTMM) endpoint.",
        REMOVE,
    );
}
