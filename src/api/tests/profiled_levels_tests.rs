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

/// Every estimator outside `foce`/`focei`/`laplace`, alone or as a chain step. The message
/// names what it saw, lists the three methods (not #1065's two) and never claims a
/// fallback.
#[test]
fn profile_refuses_every_method_outside_foce_focei_and_laplace() {
    for m in [
        EstimationMethod::Saem,
        EstimationMethod::Imp,
        EstimationMethod::Impmap,
        EstimationMethod::FoceGn,
        EstimationMethod::FoceGnHybrid,
        EstimationMethod::Bayes,
        EstimationMethod::Vi,
    ] {
        let alone = FitOptions {
            method: m,
            ..FitOptions::default()
        };
        let chain = FitOptions {
            methods: vec![EstimationMethod::Laplace, m],
            ..FitOptions::default()
        };
        for opts in [alone, chain] {
            let msg = one_refusal(
                &scope(&model_of(&counted()), &opts),
                &format!(
                    "which only `foce`, `focei` and `laplace` estimate; this fit runs `{}`.",
                    m.label()
                ),
                REMOVE,
            );
            assert!(!msg.contains("only `foce` and `focei`"), "{msg}");
            assert!(!msg.contains("fall"), "{msg}");
        }
    }
}

/// The accepted side of the gate (#2107): `foce`, `focei` and `laplace` at one node, alone
/// and as a chain; and `foce` with `n_agq > 1`, which FOCE ignores (`agq_nodes_for` is
/// `None`) and which was refused only because `n_agq` used to be read fit-wide.
#[test]
fn profile_accepts_foce_focei_and_laplace_at_one_node() {
    for ok in [
        EstimationMethod::Foce,
        EstimationMethod::FoceI,
        EstimationMethod::Laplace,
    ] {
        let opts = FitOptions {
            method: ok,
            ..FitOptions::default()
        };
        assert!(scope(&model_of(&counted()), &opts).is_empty(), "{ok:?}");
    }
    let chain = FitOptions {
        methods: vec![EstimationMethod::FoceI, EstimationMethod::Laplace],
        ..FitOptions::default()
    };
    assert!(scope(&model_of(&counted()), &chain).is_empty());
    let foce_agq = FitOptions {
        method: EstimationMethod::Foce,
        n_agq: 3,
        ..FitOptions::default()
    };
    let msgs = scope(&model_of(&counted()), &foce_agq);
    assert!(msgs.is_empty(), "{msgs:#?}");
}

/// Adaptive quadrature is accepted (#2107 PR-A): `laplace` and `focei` at `n_agq = 3`, alone
/// and as a `[focei, laplace]` chain, in which both stages run quadrature (`fit_inner`
/// keeps the chain's `n_agq` on each stage). The control of PR-L's per-stage refusal: dies
/// when any `n_agq > 1` stage is refused again.
#[test]
fn profile_accepts_adaptive_quadrature_under_laplace_and_focei() {
    for m in [EstimationMethod::Laplace, EstimationMethod::FoceI] {
        let opts = FitOptions {
            method: m,
            n_agq: 3,
            ..FitOptions::default()
        };
        assert_eq!(opts.agq_nodes(), Some(3), "{m:?} runs quadrature");
        let msgs = scope(&model_of(&counted()), &opts);
        assert!(msgs.is_empty(), "{m:?}: {msgs:#?}");
    }
    let chain = FitOptions {
        methods: vec![EstimationMethod::FoceI, EstimationMethod::Laplace],
        n_agq: 3,
        ..FitOptions::default()
    };
    let msgs = scope(&model_of(&counted()), &chain);
    assert!(msgs.is_empty(), "{msgs:#?}");
}

/// A grid the quadrature cannot evaluate is refused for a profiled block as for any other
/// fit, by `E_AGQ_GRID_TOO_LARGE` and not by the profile's scope: `21⁴ = 194,481` nodes
/// over four random effects is above `MAX_AGQ_GRID`, so the profile solve never sees it.
/// The same model at `n_agq = 3` (`81` nodes) is the straddle, accepted by both checks.
#[test]
fn a_profiled_block_over_the_agq_grid_limit_is_refused_by_the_grid_check() {
    let wide = counted()
        .replace(
            "  omega ETA_E0 ~ 4.0\n",
            "  omega ETA_E0 ~ 4.0\n  omega ETA_A ~ 1.0\n  omega ETA_B ~ 1.0\n  \
             omega ETA_C ~ 1.0\n",
        )
        .replace("+ ETA_E0\n", "+ ETA_E0 + ETA_A + ETA_B + ETA_C\n");
    let model = model_of(&wide);
    assert_eq!(model.n_eta, 4);
    let codes = |n_agq: usize| -> Vec<String> {
        let opts = FitOptions {
            method: EstimationMethod::Laplace,
            n_agq,
            ..FitOptions::default()
        };
        crate::api::check_model_options(&model, &opts)
            .into_iter()
            .filter(|d| d.code.starts_with("E_"))
            .map(|d| d.code)
            .collect()
    };
    assert!(
        crate::estimation::agq::grid_size(21, 4) > crate::estimation::agq::MAX_AGQ_GRID
            && crate::estimation::agq::grid_size(3, 4) <= crate::estimation::agq::MAX_AGQ_GRID
    );
    assert_eq!(codes(21), vec!["E_AGQ_GRID_TOO_LARGE".to_string()]);
    assert!(codes(3).is_empty(), "{:?}", codes(3));
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

/// A `correlation` line of every kind is in scope (#2110): the profile solve reads the
/// objective's own dense `R`. Before #2110 each was refused with "not yet supported".
#[test]
fn profile_accepts_a_correlation_line() {
    for kind in ["cs", "ar1", "car1"] {
        let text = counted()
            .replace(
                "  omega ETA_E0",
                "  theta RHO(0.3, 0.0, 0.95)\n  omega ETA_E0",
            )
            .replace(
                "  STUDY categorical",
                "  STUDY categorical\n  ARM categorical",
            )
            .replace(
                "DV ~ additive(ADD_ERR)",
                &format!("DV ~ additive(ADD_ERR)\n  correlation {kind}(RHO) by ARM"),
            );
        let model = model_of(&text);
        assert!(model.residual_serial.is_some(), "{kind}: the line parsed");
        let msgs = scope(&model, &FitOptions::default());
        assert!(msgs.is_empty(), "{kind}: {msgs:#?}");
    }
}

/// The correlation line's own refusals stay, and `fit()` raises them **before** the
/// profile's scope check (#2110): profiled F1 with `cs(RHO) by OCC` over the arms, where
/// `KAPPA_ARM` already lives, is refused as confounded — also when the fit asks for a
/// `global_search`, which the profile scope refuses on its own. Dies when the data check
/// is moved after `check_model_options`, or when the confounded check skips a profiled
/// model.
#[test]
fn a_confounded_correlation_is_refused_before_the_profile_scope() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(root.join("tests/data/profile_mbma/f1_profile.ferx"))
        .unwrap()
        .replace(
            "  omega ETA_E0",
            "  theta RHO(0.3, 0.0, 0.95)\n  omega ETA_E0",
        )
        .replace(
            "DV ~ additive(ADD_ERR) weight = SE",
            "DV ~ additive(ADD_ERR) weight = SE\n  correlation cs(RHO) by OCC",
        );
    assert!(text.contains("correlation cs(RHO)"), "fixture moved");
    let path = std::env::temp_dir().join(format!(
        "ferx_profile_confounded_{}.ferx",
        std::process::id()
    ));
    std::fs::write(&path, text).unwrap();
    let p = crate::api::prepare_run(
        path.to_str().unwrap(),
        Some(
            root.join("tests/data/mbma_placebo/mbma_placebo.csv")
                .to_str()
                .unwrap(),
        ),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let _ = std::fs::remove_file(&path);
    for (how, opts) in [
        ("default", p.parsed.fit_options.clone()),
        (
            "with a `global_search`",
            FitOptions {
                global_search: true,
                ..p.parsed.fit_options.clone()
            },
        ),
    ] {
        let opts = FitOptions {
            verbose: false,
            outer_maxiter: 0,
            ..opts
        };
        let err = crate::api::fit(&p.parsed.model, &p.population, &p.init_params, &opts)
            .err()
            .unwrap_or_else(|| panic!("{how}: fitted"));
        assert_eq!(
            err.code(),
            Some("E_RESIDUAL_CORRELATION_CONFOUNDED_KAPPA"),
            "{how}: {err}"
        );
    }
    // The straddle: the `global_search` request alone is the profile scope's refusal.
    let g = FitOptions {
        global_search: true,
        ..p.parsed.fit_options.clone()
    };
    assert_eq!(scope(&p.parsed.model, &g).len(), 1);
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

/// Every covariance estimator is in scope (#2108), requested explicitly with the step on:
/// `r`, and `s` and `rsr`, which #2103 refused with "computes `r` only for now".
#[test]
fn profile_accepts_every_covariance_method() {
    for m in [
        CovarianceMethod::Hessian,
        CovarianceMethod::CrossProduct,
        CovarianceMethod::Sandwich,
    ] {
        let opts = FitOptions {
            run_covariance_step: true,
            covariance_method: m,
            covariance_method_set: true,
            ..FitOptions::default()
        };
        let msgs = scope(&model_of(&counted()), &opts);
        assert!(msgs.is_empty(), "{m:?}: {msgs:#?}");
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

// ── W_COV_RSR_ONE_SUBJECT_LEVELS: an `rsr` level SE no data informs ────────────────

/// F1 (`mbma_placebo`, one subject per study) on its data, profiled or not, with
/// `edit_csv` applied to the data text.
fn f1_prepared(profiled: bool, edit_csv: impl Fn(String) -> String) -> crate::api::PreparedRun {
    f1_prepared_with(profiled, |t| t, edit_csv)
}

/// [`f1_prepared`] with `edit_model` applied to the model text first.
fn f1_prepared_with(
    profiled: bool,
    edit_model: impl Fn(String) -> String,
    edit_csv: impl Fn(String) -> String,
) -> crate::api::PreparedRun {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let text =
        std::fs::read_to_string(root.join("tests/data/profile_mbma/f1_profile.ferx")).unwrap();
    let block = "](0.0) profile";
    assert!(text.contains(block), "fixture moved");
    let text = edit_model(text);
    let text = if profiled {
        text
    } else {
        text.replace(block, "](0.0, -50.0, 50.0)")
    };
    let csv = edit_csv(
        std::fs::read_to_string(root.join("tests/data/mbma_placebo/mbma_placebo.csv")).unwrap(),
    );
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir();
    let (m, d) = (
        dir.join(format!("ferx_one_subject_{}_{n}.ferx", std::process::id())),
        dir.join(format!("ferx_one_subject_{}_{n}.csv", std::process::id())),
    );
    std::fs::write(&m, text).unwrap();
    std::fs::write(&d, csv).unwrap();
    let p = crate::api::prepare_run(m.to_str().unwrap(), Some(d.to_str().unwrap()))
        .unwrap_or_else(|e| panic!("{e}"));
    let _ = (std::fs::remove_file(&m), std::fs::remove_file(&d));
    p
}

/// Study 1's rows again under a second subject `ID 99`, so study 1's levels are read by
/// two subjects and every other study's by one.
fn study_1_twice(csv: String) -> String {
    let extra: String = csv
        .lines()
        .skip(1)
        .filter(|l| l.starts_with("1,1,"))
        .map(|l| format!("99{}\n", &l[1..]))
        .collect();
    assert!(!extra.is_empty(), "fixture moved");
    format!("{csv}{extra}")
}

/// The note in full, on F1's own data (every level read by one subject) and with study 1
/// read twice (its levels shared, the rest single), profiled and unprofiled. Asserted by
/// equality, so deleting any sentence of it fails, and the two counts it prints differ in
/// the second case, so swapping them fails too. The gate, from both sides in one test:
/// only `rsr` carries it, never `r` or `s`. It classifies as a covariance-step warning, so
/// `run_covariance` replaces it with the step it reruns.
#[test]
fn rsr_names_the_levels_a_single_subject_reads() {
    for profiled in [true, false] {
        let p = f1_prepared(profiled, |c| c);
        let labels = p.parsed.model.theta_blocks().level_blocks()[0]
            .labels()
            .to_vec();
        let n = labels.len();
        let want = |k: usize, e: &str| {
            format!(
                "W_COV_RSR_ONE_SUBJECT_LEVELS: under covariance_method = rsr, {k} of the {n} \
                 levels of theta PLACEBO[...] (e.g. {e}) are each read by a single subject. At \
                 the estimate such a level's own score is zero or nearly so, so the sandwich \
                 carries almost none of its variability and its standard error is not usable. \
                 Read those levels' standard errors from covariance_method = r."
            )
        };
        let at = |p: &crate::api::PreparedRun, m: Option<CovarianceMethod>| {
            crate::api::rsr_one_subject_levels_warning(&p.parsed.model, &p.population, m)
        };
        let got = at(&p, Some(CovarianceMethod::Sandwich)).expect("rsr carries the note");
        assert_eq!(got, want(n, &labels[0]), "profiled={profiled}");
        for silent in [
            Some(CovarianceMethod::Hessian),
            Some(CovarianceMethod::CrossProduct),
            None,
        ] {
            assert_eq!(at(&p, silent), None, "profiled={profiled}, {silent:?}");
        }
        let entry = crate::types::classify_warning(&got);
        assert_eq!(entry.category, crate::types::WarningCode::CovarianceStep);
        assert_eq!(entry.severity, crate::types::WarningSeverity::Warning);

        let shared = f1_prepared(profiled, study_1_twice);
        let study_1 = labels.iter().filter(|l| l.starts_with("STUDY=1,")).count();
        assert!(
            study_1 > 0 && study_1 < n,
            "fixture: study 1 holds some levels"
        );
        let first_single = labels.iter().find(|l| !l.starts_with("STUDY=1,")).unwrap();
        assert_eq!(
            at(&shared, Some(CovarianceMethod::Sandwich)),
            Some(want(n - study_1, first_single)),
            "profiled={profiled}, study 1 read twice"
        );
    }
}

/// Only a level that has an SE is counted (#2415 review rows 1, 2). A `FIX` block has none
/// (every level reports SE 0), so the F1 twin with its block `FIX` is silent. A
/// `contrast = ref` block's reference level is the constant 0, so F1 under `ref` names 23 of
/// its 24 levels, and the example is not the reference, which sorts first. Under `ref` the
/// block cannot sit beside the subject-level `ETA_E0` it would absorb, so that twin drops
/// it. Dies when the count reads the labels instead of the levels' rules: the `FIX` twin
/// then names all 24, and the `ref` twin names 24 with the reference as its example.
#[test]
fn rsr_counts_only_levels_that_have_a_standard_error() {
    let at = |p: &crate::api::PreparedRun| {
        crate::api::rsr_one_subject_levels_warning(
            &p.parsed.model,
            &p.population,
            Some(CovarianceMethod::Sandwich),
        )
    };
    let fix = f1_prepared_with(true, |t| t.replace("](0.0) profile", "](0.0) FIX"), |c| c);
    assert!(
        fix.parsed
            .model
            .default_params
            .theta_fixed
            .iter()
            .any(|&f| f),
        "fixture: the block is FIX"
    );
    assert_eq!(at(&fix), None, "a FIX block has no rsr SE to call unusable");

    let to_ref = |t: String| {
        let edited = t
            .replace("contrast = sum_to_zero_within", "contrast = ref")
            .replace("  omega ETA_E0 ~ 4.0 (sd)\n", "")
            .replace(" + ETA_E0", "");
        assert_ne!(edited, t, "F1 moved");
        edited
    };
    for profiled in [true, false] {
        let p = f1_prepared_with(profiled, to_ref, |c| c);
        let labels = p.parsed.model.theta_blocks().level_blocks()[0]
            .labels()
            .to_vec();
        assert_eq!(labels.len(), 24);
        let got = at(&p).unwrap_or_else(|| panic!("profiled={profiled}: no note"));
        assert!(
            got.contains(&format!(
                "23 of the 24 levels of theta PLACEBO[...] (e.g. {}) are each read",
                labels[1]
            )),
            "profiled={profiled}: {got}"
        );
        assert!(
            !got.contains(&format!("(e.g. {})", labels[0])),
            "the reference level is not the example: {got}"
        );
    }
}

/// Every subject's rows duplicated under a second subject, except the rows at one (study,
/// visit), so exactly one level, `STUDY=1,TIME=0`, is read by a single subject.
fn all_but_one_level_twice(csv: String) -> String {
    let extra: String = csv
        .lines()
        .skip(1)
        .filter(|l| {
            let f: Vec<&str> = l.split(',').collect();
            !(f[1] == "1" && f[3] == "0")
        })
        .map(|l| {
            let (id, rest) = l.split_once(',').unwrap();
            format!("{},{rest}\n", id.parse::<u32>().unwrap() + 100)
        })
        .collect();
    assert!(!extra.is_empty(), "fixture moved");
    format!("{csv}{extra}")
}

/// One single level reads in the singular and names it without "e.g." (#2415 review row 4):
/// "1 of the 24 levels … (STUDY=1,TIME=0) is read by a single subject". Dies when the
/// one-level arm is dropped, which prints "1 of the 24 levels (e.g. …) are each read".
#[test]
fn rsr_names_a_single_one_subject_level_in_the_singular() {
    for profiled in [true, false] {
        let p = f1_prepared(profiled, all_but_one_level_twice);
        let got = crate::api::rsr_one_subject_levels_warning(
            &p.parsed.model,
            &p.population,
            Some(CovarianceMethod::Sandwich),
        )
        .unwrap_or_else(|| panic!("profiled={profiled}: no note"));
        assert!(
            got.starts_with(
                "W_COV_RSR_ONE_SUBJECT_LEVELS: under covariance_method = rsr, 1 of the 24 \
                 levels of theta PLACEBO[...] (STUDY=1,TIME=0) is read by a single subject. "
            ),
            "profiled={profiled}: {got}"
        );
    }
}

/// A block every level of which several subjects read carries no note: S1's placebo model
/// with a visit block `PLACEBO[TIME]` on its 100 subjects. Dies when the count admits a
/// level read by more than one subject.
#[test]
fn rsr_is_silent_when_every_level_is_shared() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let text =
        std::fs::read_to_string(root.join("tests/data/profile_mbma/s1_profile.ferx")).unwrap();
    let block = "theta PLACEBO[STUDY, TIME](0.0) profile";
    assert!(text.contains(block), "fixture moved");
    let path = std::env::temp_dir().join(format!(
        "ferx_one_subject_visit_{}.ferx",
        std::process::id()
    ));
    std::fs::write(
        &path,
        text.replace(block, "theta PLACEBO[TIME](0.0) profile"),
    )
    .unwrap();
    let p = crate::api::prepare_run(
        path.to_str().unwrap(),
        Some(
            root.join("tests/data/profile_mbma/s100t8.csv")
                .to_str()
                .unwrap(),
        ),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        crate::api::rsr_one_subject_levels_warning(
            &p.parsed.model,
            &p.population,
            Some(CovarianceMethod::Sandwich)
        ),
        None
    );
}
