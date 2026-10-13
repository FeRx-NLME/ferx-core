//! Validation of a `correlation cs|ar1(RHO) by COL` line (#1081): the phase-1 scope
//! (`E_RESIDUAL_CORRELATION_SCOPE`), the data-dependent range of ρ
//! (`E_RESIDUAL_CORRELATION_RANGE`), tied times under `ar1`
//! (`E_RESIDUAL_CORRELATION_TIES`) and a line that couples nothing
//! (`W_RESIDUAL_CORRELATION_INERT`). Every sentence of every message is asserted, and
//! every gate from both sides in one test.

use super::{check_model_data, check_model_data_warnings, check_model_options};
use crate::diagnostics::Diagnostic;
use crate::parser::model_parser::parse_full_model;
use crate::types::{CompiledModel, FitOptions, Population, Subject};
use std::collections::HashMap;

/// A one-compartment IV model with a combined error and a `correlation` line.
/// `rho` is the whole θ declaration tail, e.g. `(0.3, -0.4, 0.95)` or `(0.5) FIX`.
fn src(kind: &str, rho: &str, extra_params: &str, extra_fit: &str) -> String {
    format!(
        "[parameters]\n  theta TVCL(1.0, 0.01, 10.0)\n  theta TVV(10.0, 0.1, 100.0)\n  \
         theta RHO{rho}\n  omega ETA_CL ~ 0.09\n{extra_params}  sigma PROP_ERR ~ 0.2 (sd)\n  \
         sigma ADD_ERR ~ 1.0 (sd)\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  \
         V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  \
         DV ~ combined(PROP_ERR, ADD_ERR)\n  correlation {kind}(RHO) by ARM\n[fit_options]\n  \
         method = focei\n{extra_fit}"
    )
}

fn model(kind: &str, rho: &str) -> CompiledModel {
    parse_full_model(&src(kind, rho, "", ""))
        .expect("parse")
        .model
}

/// One subject per `arms` entry; each row is `(ARM, TIME)`.
fn population(subjects: &[&[(f64, f64)]]) -> Population {
    let subjects = subjects
        .iter()
        .enumerate()
        .map(|(i, rows)| Subject {
            id: format!("S{}", i + 1),
            obs_times: rows.iter().map(|r| r.1).collect(),
            observations: vec![1.0; rows.len()],
            obs_cmts: vec![1; rows.len()],
            cens: vec![0; rows.len()],
            covariates: HashMap::from([("ARM".to_string(), rows[0].0)]),
            obs_covariates: rows
                .iter()
                .map(|r| HashMap::from([("ARM".to_string(), r.0)]))
                .collect(),
            ..Default::default()
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec!["ARM".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn with_code<'a>(diags: &'a [Diagnostic], code: &str) -> Vec<&'a Diagnostic> {
    diags.iter().filter(|d| d.code == code).collect()
}

fn one<'a>(diags: &'a [Diagnostic], code: &str) -> &'a str {
    let hits = with_code(diags, code);
    assert_eq!(hits.len(), 1, "{code}: {diags:#?}");
    &hits[0].message
}

// ── scope ───────────────────────────────────────────────────────────────────

fn scope_for(model_src: &str, opts: &FitOptions) -> Vec<Diagnostic> {
    let parsed = parse_full_model(model_src).expect("parse");
    check_model_options(&parsed.model, opts)
}

fn focei() -> FitOptions {
    FitOptions {
        method: crate::types::EstimationMethod::FoceI,
        ..FitOptions::default()
    }
}

/// The method gate, both sides: `foce` and `focei` pass, `saem` (and a chain that
/// reaches it) is refused, naming the method and the supported set.
#[test]
fn scope_names_an_unsupported_method_and_the_supported_set() {
    let s = src("cs", "(0.3, -0.4, 0.95)", "", "");
    for m in [
        crate::types::EstimationMethod::Foce,
        crate::types::EstimationMethod::FoceI,
    ] {
        let opts = FitOptions {
            method: m,
            ..FitOptions::default()
        };
        assert!(with_code(&scope_for(&s, &opts), "E_RESIDUAL_CORRELATION_SCOPE").is_empty());
    }
    let saem = FitOptions {
        method: crate::types::EstimationMethod::Saem,
        ..FitOptions::default()
    };
    let d = scope_for(&s, &saem);
    let msg = one(&d, "E_RESIDUAL_CORRELATION_SCOPE");
    assert!(
        msg.contains("`correlation cs(RHO) by ARM` is not supported with method = saem."),
        "{msg}"
    );
    assert!(
        msg.contains("A residual correlation is fitted by method = foce and focei only."),
        "{msg}"
    );
    // The block_sigma refusal must not speak for a `correlation` line.
    assert!(
        d.iter().all(|x| !x.code.starts_with("E_BLOCK_SIGMA")),
        "{d:#?}"
    );
    // A chain that ends in a supported method but passes through saem is refused too.
    let chain = FitOptions {
        methods: vec![
            crate::types::EstimationMethod::Saem,
            crate::types::EstimationMethod::FoceI,
        ],
        ..FitOptions::default()
    };
    assert!(one(&scope_for(&s, &chain), "E_RESIDUAL_CORRELATION_SCOPE").contains("saem"));
}

/// `n_agq > 1` turns FOCEI into adaptive quadrature, which is out of scope.
#[test]
fn scope_refuses_quadrature() {
    let s = src("ar1", "(0.3, -0.4, 0.95)", "", "");
    let agq = FitOptions {
        n_agq: 3,
        ..focei()
    };
    let msg = one(&scope_for(&s, &agq), "E_RESIDUAL_CORRELATION_SCOPE").to_string();
    assert!(
        msg.contains("`correlation ar1(RHO) by ARM` is not supported with n_agq = 3."),
        "{msg}"
    );
    assert!(with_code(&scope_for(&s, &focei()), "E_RESIDUAL_CORRELATION_SCOPE").is_empty());
}

/// κ: refused, and the message says why the two are not combined — the measured
/// confounding of CS (ρ ≥ 0) with a weight-loaded arm κ.
#[test]
fn scope_refuses_kappa_and_says_it_is_the_same_covariance() {
    let s = src(
        "cs",
        "(0.3, -0.4, 0.95)",
        "  kappa K_CL ~ 0.04\n",
        "  iov_column = OCC\n",
    )
    .replace("CL = TVCL * exp(ETA_CL)", "CL = TVCL * exp(ETA_CL + K_CL)");
    let msg = one(&scope_for(&s, &focei()), "E_RESIDUAL_CORRELATION_SCOPE").to_string();
    assert!(
        msg.contains("is not supported with inter-occasion random effects (kappa)."),
        "{msg}"
    );
    assert!(
        msg.contains(
            "A kappa intercept loaded by the residual weight on the same grouping is the same \
             covariance as `cs` with ρ ≥ 0, so the two are not identifiable together"
        ),
        "{msg}"
    );
    assert!(msg.contains("fit one or the other (#1082)."), "{msg}");
    // The confounding sentence belongs to κ only.
    let saem = FitOptions {
        method: crate::types::EstimationMethod::Saem,
        ..FitOptions::default()
    };
    let other = scope_for(&src("cs", "(0.3, -0.4, 0.95)", "", ""), &saem);
    assert!(!one(&other, "E_RESIDUAL_CORRELATION_SCOPE").contains("identifiable"));
}

/// M3 censoring and an SDE model each get their own refusal.
#[test]
fn scope_refuses_m3_and_sde() {
    let m3 = src("cs", "(0.3, -0.4, 0.95)", "", "  bloq_method = m3\n");
    assert!(
        one(&scope_for(&m3, &focei()), "E_RESIDUAL_CORRELATION_SCOPE")
            .contains("is not supported with bloq_method = m3.")
    );
    let sde = src("cs", "(0.3, -0.4, 0.95)", "", "")
        .replace("pk one_cpt_iv(cl=CL, v=V)", "ode(obs_cmt=central, states=[central])")
        .replace(
            "[error_model]",
            "[odes]\n  d/dt(central) = -(CL / V) * central\n[diffusion]\n  central ~ 0.01\n[error_model]",
        );
    assert!(
        one(&scope_for(&sde, &focei()), "E_RESIDUAL_CORRELATION_SCOPE")
            .contains("is not supported with an SDE ([diffusion]) model.")
    );
}

// ── range ───────────────────────────────────────────────────────────────────

/// Compound symmetry over the largest block: three rows ⇒ ρ > −1/2. A lower bound
/// at −0.6 is refused with the block, its size and the interval; one at −0.4 is
/// accepted — the same data, one number apart.
#[test]
fn cs_range_names_the_largest_block_and_its_interval() {
    let pop = population(&[
        &[(1.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        &[(7.0, 0.0), (7.0, 1.0), (7.0, 2.0), (3.0, 4.0)],
    ]);
    let refused = check_model_data(&model("cs", "(0.3, -0.6, 0.95)"), &pop);
    let msg = one(&refused, "E_RESIDUAL_CORRELATION_RANGE");
    assert!(
        msg.contains("`correlation cs(RHO) by ARM`: the lower bound of theta RHO is -0.6"),
        "{msg}"
    );
    assert!(
        msg.contains("but subject S2 has 3 rows in the block ARM=7"),
        "{msg}"
    );
    assert!(
        msg.contains(
            "compound symmetry over 3 rows is a valid covariance only for ρ in (−1/(3−1), 1) = \
             (-0.5000, 1)"
        ),
        "{msg}"
    );
    assert!(msg.contains("Keep ρ above -0.5000."), "{msg}");
    let accepted = check_model_data(&model("cs", "(0.3, -0.4, 0.95)"), &pop);
    assert!(
        with_code(&accepted, "E_RESIDUAL_CORRELATION_RANGE").is_empty(),
        "{accepted:#?}"
    );
    // The CS floor is a CS rule: `ar1` at −0.6 is a valid correlation.
    let ar1 = check_model_data(&model("ar1", "(0.3, -0.6, 0.95)"), &pop);
    assert!(
        with_code(&ar1, "E_RESIDUAL_CORRELATION_RANGE").is_empty(),
        "{ar1:#?}"
    );
}

/// A `FIX`ed ρ is checked at its value, and says so.
#[test]
fn cs_range_checks_a_fixed_value() {
    let pop = population(&[&[(1.0, 0.0), (1.0, 1.0), (1.0, 2.0)]]);
    let msg = one(
        &check_model_data(&model("cs", "(-0.6) FIX"), &pop),
        "E_RESIDUAL_CORRELATION_RANGE",
    )
    .to_string();
    assert!(
        msg.contains("theta RHO is fixed at -0.6, but subject S1"),
        "{msg}"
    );
    let msg = one(
        &check_model_data(&model("ar1", "(1.0) FIX"), &pop),
        "E_RESIDUAL_CORRELATION_RANGE",
    )
    .to_string();
    assert!(
        msg.contains(
            "`correlation ar1(RHO) by ARM`: theta RHO is fixed at 1; a correlation must lie \
             strictly between −1 and 1."
        ),
        "{msg}"
    );
}

/// The rail binds a free ρ only: an upper bound of 0.999 is refused free and
/// accepted `FIX`ed at the same number; a lower bound of −0.999 under `ar1` the same.
#[test]
fn the_rail_binds_a_free_rho_and_fix_lifts_it() {
    let pop = population(&[&[(1.0, 0.0), (1.0, 1.0)]]);
    let msg = one(
        &check_model_data(&model("cs", "(0.3, -0.4, 0.999)"), &pop),
        "E_RESIDUAL_CORRELATION_RANGE",
    )
    .to_string();
    assert!(
        msg.contains("`correlation cs(RHO) by ARM`: the upper bound of theta RHO is 0.999."),
        "{msg}"
    );
    assert!(
        msg.contains(
            "A free correlation is kept within ±0.995, because as |ρ| approaches 1 the rows of a \
             block become collinear and R numerically singular."
        ),
        "{msg}"
    );
    assert!(
        msg.contains("Bound theta RHO within ±0.995, or FIX it to hold a value closer to ±1."),
        "{msg}"
    );
    let fixed = check_model_data(&model("cs", "(0.999) FIX"), &pop);
    assert!(
        with_code(&fixed, "E_RESIDUAL_CORRELATION_RANGE").is_empty(),
        "{fixed:#?}"
    );
    let lower = check_model_data(&model("ar1", "(0.3, -0.999, 0.9)"), &pop);
    assert!(one(&lower, "E_RESIDUAL_CORRELATION_RANGE")
        .contains("the lower bound of theta RHO is -0.999."));
    // An unbounded θ keeps the default upper bound, which the rail refuses.
    let unbounded = check_model_data(&model("ar1", "(0.3)"), &pop);
    assert!(one(&unbounded, "E_RESIDUAL_CORRELATION_RANGE").contains("the upper bound"));
}

// ── ties ────────────────────────────────────────────────────────────────────

/// `ar1` and two rows of one block at one time: refused with the subject, the time
/// and the block. Both sides: the same tie split across two blocks, and the same tie
/// under `cs`, are accepted.
#[test]
fn ar1_refuses_tied_times_within_a_block() {
    let tied = population(&[&[(1.0, 0.0), (1.0, 2.0), (1.0, 2.0)]]);
    let msg = one(
        &check_model_data(&model("ar1", "(0.3, -0.4, 0.95)"), &tied),
        "E_RESIDUAL_CORRELATION_TIES",
    )
    .to_string();
    assert!(
        msg.contains(
            "`correlation ar1(RHO) by ARM`: subject S1 has two rows at TIME 2 in the block ARM=1."
        ),
        "{msg}"
    );
    assert!(
        msg.contains(
            "`ar1` orders a block's rows by TIME, so tied rows have no order and no lag between \
             them."
        ),
        "{msg}"
    );
    assert!(
        msg.contains("Give them distinct times, or use `cs`."),
        "{msg}"
    );
    let split = population(&[&[(1.0, 0.0), (1.0, 2.0), (2.0, 2.0)]]);
    let ok = check_model_data(&model("ar1", "(0.3, -0.4, 0.95)"), &split);
    assert!(
        with_code(&ok, "E_RESIDUAL_CORRELATION_TIES").is_empty(),
        "{ok:#?}"
    );
    let cs = check_model_data(&model("cs", "(0.3, -0.4, 0.95)"), &tied);
    assert!(
        with_code(&cs, "E_RESIDUAL_CORRELATION_TIES").is_empty(),
        "{cs:#?}"
    );
}

// ── inert ───────────────────────────────────────────────────────────────────

/// Every block a singleton: ρ is not in the likelihood. One shared block anywhere
/// in the population and the warning goes.
#[test]
fn a_line_that_couples_nothing_is_reported() {
    let m = model("cs", "(0.3, -0.4, 0.95)");
    let singles = population(&[&[(1.0, 0.0), (2.0, 1.0)], &[(1.0, 0.0)]]);
    let d = check_model_data_warnings(&m, &singles, &m.default_params);
    let msg = one(&d, "W_RESIDUAL_CORRELATION_INERT");
    assert!(
        msg.contains(
            "`correlation cs(RHO) by ARM`: no block holds more than one observation (one row per \
             group), so no two rows are correlated and ρ does not enter the likelihood"
        ),
        "{msg}"
    );
    assert!(
        msg.contains("it cannot be estimated and stays at its initial value."),
        "{msg}"
    );
    assert!(
        msg.contains("Check that the `by` column(s) group the rows you meant to correlate."),
        "{msg}"
    );
    let paired = population(&[&[(1.0, 0.0), (2.0, 1.0)], &[(1.0, 0.0), (1.0, 1.0)]]);
    let d = check_model_data_warnings(&m, &paired, &m.default_params);
    assert!(
        with_code(&d, "W_RESIDUAL_CORRELATION_INERT").is_empty(),
        "{d:#?}"
    );
}

/// FREM and a mixture each get their own refusal; the plain model gets neither.
#[test]
fn scope_refuses_frem_and_mixture() {
    let base = src("cs", "(0.3, -0.4, 0.95)", "", "");
    assert!(with_code(&scope_for(&base, &focei()), "E_RESIDUAL_CORRELATION_SCOPE").is_empty());
    let frem = src(
        "cs",
        "(0.3, -0.4, 0.95)",
        "",
        "  frem_predictions = TVV/ETA_CL:100\n  frem_sigma = ADD_ERR\n",
    );
    assert!(
        one(&scope_for(&frem, &focei()), "E_RESIDUAL_CORRELATION_SCOPE")
            .contains("is not supported with FREM covariate pseudo-observations.")
    );
    let mixture = base
        .replace(
            "  omega ETA_CL ~ 0.09\n",
            "  omega ETA_CL ~ 0.09\n  theta MIXL(0.0, -10.0, 10.0)\n",
        )
        .replace(
            "[individual_parameters]",
            "[mixture]\n  nsub = 2\n  logit(1) = MIXL\n  omega(2) ETA_CL ~ 0.4\n[individual_parameters]",
        );
    assert!(one(
        &scope_for(&mixture, &focei()),
        "E_RESIDUAL_CORRELATION_SCOPE"
    )
    .contains("is not supported with a mixture model."));
}

/// A non-Gaussian endpoint (`[binary_model]` needs the `survival` feature, which the
/// TTE/CTMM CI leg builds).
#[cfg(feature = "survival")]
#[test]
fn scope_refuses_a_non_gaussian_endpoint() {
    let base = src("cs", "(0.3, -0.4, 0.95)", "", "");
    let binary = base.replace(
        "[fit_options]",
        "[binary_model]\n  cmt   = 3\n  logit = TVCL - 1\n[fit_options]",
    );
    assert!(one(
        &scope_for(&binary, &focei()),
        "E_RESIDUAL_CORRELATION_SCOPE"
    )
    .contains("is not supported with a time-to-event, binary, categorical or Markov endpoint."));
}
