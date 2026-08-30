//! Does VI recover the **between-subject** variance `Ω` on a deep compartment model?
//!
//!   cargo test --release --features nn,slow-tests --test vi_dcm_omega_recovery
//!
//! ## Why this file exists
//!
//! `VI_PLAN.md` §6 item 13 asked for a simulated-DCM recovery test and it never landed.
//! The gap was specific and consequential: **every** `Ω` assertion VI carries is blind to
//! a downward bias in `Ω_bsv` by construction.
//!
//! - The exact-case oracle (`vi/elbo_oracle.rs`) fits a linear-Gaussian model, where
//!   full-rank VI is *exact*. Collapse arises from `q` underfitting a non-Gaussian
//!   posterior, which the oracle removes by design.
//! - `closed_form_omega_matches_the_exact_posterior_moments`,
//!   `closed_form_omega_is_a_stationary_point_of_the_kl` and
//!   `closed_form_omega_zeroes_the_omega_gradient` all check that
//!   `Ω* = (1/N)Σ(Sᵢ+μᵢμᵢᵀ)` is the right maximizer *given* the `φᵢ`. Collapse is a bias
//!   **in** the `φᵢ`, so a correct maximizer reports it faithfully.
//! - `both_omega_update_routes_agree_approximately` compares the two `Ω` routes at a
//!   relative tolerance of `1.0` — a 2× disagreement passes.
//! - `tests/vi_time_varying_iov.rs` bands `Ω_iov` but simulated `Ω_bsv` and never looked
//!   at it. (Fixed in the same change as this file.)
//!
//! ## What went wrong on real data, and what it taught this file
//!
//! A busulfan-shaped DCM fit (60 subjects, `[covariate_nn]` over `WT` and `CRCL`,
//! `layers = [6, 6]` ⇒ 74 weights, IOV on CL) returned `ω²(CL) = 0.0173` against a
//! simulated truth of `0.09` — **5× low** — while reporting `converged: true`,
//! `elbo_tightness_ratio: 0.544` and `n_fd_subjects: 0`. Every health check
//! `docs/estimation/vi.qmd` tells a user to read came back green.
//!
//! A capacity sweep on that exact dataset (same seed, same options, only `layers`
//! changing) isolated the cause:
//!
//! | `layers` | NN weights | ω²(CL) | −2 log L |
//! |---|---|---|---|
//! | `[2]` | 12 | 0.1323 | 1151.3 |
//! | `[3]` | 17 | 0.1318 | 1147.6 |
//! | `[6, 6]` | 74 | **0.0173** | 1061.0 |
//!
//! The dataset has 60 subjects and **60 distinct `(WT, CRCL)` pairs**. A 74-weight
//! network over two subject-unique continuous covariates can index subjects, so
//! between-subject clearance is explainable as a covariate effect and `Ω` has nothing
//! left to hold. VI's Adam finds that optimum (87 OFV points better than FOCEI); FOCEI's
//! outer loop does not reach it, which is why FOCEI looked correct there by accident.
//!
//! ## The two mechanisms, and why the design decides which one you get
//!
//! 1. **Variational variance understatement.** `q` is a Gaussian approximation to a
//!    non-Gaussian posterior and understates its spread; `Ω*` averages `S + μμᵀ`, so the
//!    understatement passes through. Documented (`docs/estimation/vi.qmd`, VI_PLAN §4.4,
//!    §10.6), expected, and mild — it averages over `N` subjects.
//! 2. **The network absorbing between-subject variation.** A flexible fixed-effects model
//!    can explain with `θ` what belongs to `η`. This one is *not* mild, is not specific to
//!    VI, and scales with network capacity rather than washing out with `N`.
//!
//! (2) is what `NN_REGULARIZATION_PLAN.md` exists to address and what the paper's
//! Discussion asks for. Crucially, **(2) requires a design that lets the network separate
//! subjects**, and getting that design right took two attempts:
//!
//! - **Tied covariate levels** (12 levels × 5 subjects). `ω²(CL)` *rose* toward truth as
//!   capacity grew — `0.1207` at `[4]` → `0.0921` at `[8,8]`, truth `0.09` — the opposite
//!   sign from busulfan, because with ties the network cannot index subjects however wide
//!   it is. Kept, as [`capacity_alone_does_not_collapse_iiv_when_the_design_identifies_omega`].
//! - **One subject-unique covariate** (60 distinct `WT`). Still no collapse: `0.1256` at
//!   `[4]` → `0.1074` at `[8,8]`. 60 distinct values on a single axis ask a `tanh` net for
//!   a wiggly interpolant over an ordered domain, and its smoothness bias makes that a hard
//!   optimum for Adam to find.
//! - **Two subject-unique covariates**, scattered independently — busulfan's actual
//!   configuration. This is what the fixture uses now.
//!
//! So capacity is not the variable, and neither is subject-uniqueness on its own;
//! **capacity relative to the number of distinct covariate patterns, in a space where those
//! patterns are actually separable**, is. This file tests both designs and asserts the
//! contrast.
//!
//! ## Validation strategy
//!
//! A round-trip against ferx's own `simulate()`, not NONMEM — the same rationale as
//! `tests/vi_time_varying_iov.rs`. The question is whether the estimator recovers a known
//! `Ω`, and only a simulation knows the true `Ω`. A NONMEM comparison would establish that
//! two tools agree on an estimate, not that the estimate is right, and NONMEM has no DCM
//! to compare against in the first place.
//!
//! **The truth is an explicit analytic covariate model, and the fitted model is the DCM.**
//! That asymmetry is the point. Simulating from a DCM and fitting the same DCM would start
//! the network at (or near) the weights that generated the data, so the fit would never
//! have to learn anything and the absorption mechanism in (2) could not appear. Here the
//! network starts flat, at its declared `init`, and has to discover the `WT` relationship
//! — which is the situation a real DCM fit is in, and the situation in which it can
//! over-explain. `Ω`, `σ` and the covariate relationship are all known exactly either way,
//! which is all the assertions need. NN *weights* are never asserted: they are
//! non-identifiable under permutation symmetry (see `tests/nn_fit_convergence.rs`).

#![cfg(feature = "nn")]

use ferx_core::{
    fit, simulate_with_seed, CompiledModel, DoseEvent, EstimationMethod, FitOptions, FitResult,
    ModelParameters, Population, SimOutcome, Subject, ViFinalOfv,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Truth
// ---------------------------------------------------------------------------

const TRUE_TVCL: f64 = 1.0;
const TRUE_TVV: f64 = 20.0;
/// Allometric-ish exponents on `WT`. The DCM has to learn this shape from scratch.
const TRUE_WT_CL: f64 = 0.75;
const TRUE_WT_V: f64 = 1.0;
/// The quantity under test. SD ≈ 0.30 on log-CL — a realistic population IIV, and large
/// enough that a collapse toward zero is unambiguous rather than a Monte-Carlo wobble.
const TRUE_OMEGA_CL: f64 = 0.09;
const TRUE_OMEGA_V: f64 = 0.04;
const TRUE_SIGMA: f64 = 0.1;

const N_SUBJECTS: usize = 60;
/// 12 distinct weights, 5 subjects each — the **identified** design.
///
/// The ties are load-bearing. With 60 distinct weights, `WT` is a subject index in
/// disguise and a wide enough network can memorize each subject's individual clearance —
/// which makes `Ω` unidentifiable *in the design* rather than under-recovered *by the
/// estimator*. With five subjects sharing each weight, the typical-value function cannot
/// separate them however flexible it is, so `Ω` is properly identified and any shortfall
/// is attributable to the estimator. It is also what many real covariates look like:
/// rounded, repeated, coarser than the population.
///
/// [`CovariateDesign::SubjectUnique`] is the other half of the pair — the case real
/// busulfan data actually presented, where the ties are absent and memorization is on the
/// table.
const N_WT_LEVELS: usize = 12;
const SEED: u64 = 20_260_813;

/// Which covariate design a fixture is built on. The distinction is the subject of this
/// file: it is what decides whether network capacity can absorb `Ω`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CovariateDesign {
    /// 12 weight levels, 5 subjects each. `Ω` is identified whatever the network does.
    Tied,
    /// 60 distinct `(WT, CRCL)` pairs, one per subject. The covariate vector is a subject
    /// index in disguise, so a network with enough capacity can explain between-subject
    /// variation as a covariate effect. This is the busulfan configuration.
    SubjectUnique,
}

impl CovariateDesign {
    /// Weight for subject `i`, spanning 50–94 kg under either design.
    fn weight_of(self, i: usize) -> f64 {
        match self {
            CovariateDesign::Tied => 50.0 + 4.0 * (i % N_WT_LEVELS) as f64,
            CovariateDesign::SubjectUnique => 50.0 + 44.0 * (i as f64) / (N_SUBJECTS - 1) as f64,
        }
    }

    /// Creatinine clearance for subject `i`, spanning 45–150 mL/min.
    ///
    /// **Scattered independently of `WT`**, via a golden-ratio low-discrepancy sequence.
    /// That is load-bearing twice over. `weight_of` is monotone in `i`, so a `CRCL` that
    /// were also monotone would be a deterministic function of `WT` and the network would
    /// face a one-dimensional problem wearing two inputs — which is the fixture this
    /// replaced, and it did not reproduce the collapse: 60 distinct values on a *line*
    /// still ask a `tanh` net for a wiggly monotone-domain interpolant, and its smoothness
    /// bias makes that hard for Adam to find. Scattered in 2-D, 60 points separate easily.
    ///
    /// Deterministic rather than random so the fixture stays reproducible without
    /// threading another RNG through the design.
    fn crcl_of(self, i: usize) -> f64 {
        // Under `Tied`, `CRCL` keys off the *same* level index as `WT`, so the two move
        // together and there are exactly `N_WT_LEVELS` distinct pairs. Drawing them
        // independently would give up to 12 × 12 = 144 possible pairs for 60 subjects,
        // i.e. subject-unique in all but name, and would silently destroy the very
        // identification the `Tied` design exists to provide.
        let idx = match self {
            CovariateDesign::Tied => i % N_WT_LEVELS,
            CovariateDesign::SubjectUnique => i,
        };
        45.0 + 105.0 * (idx as f64 * 0.618_033_988_749_895).fract()
    }
}

/// The data-generating model: a power-law covariate model on `WT` and a saturating
/// (`tanh`-shaped) one on `CRCL`.
///
/// The `CRCL` term is busulfan's, `1 + 0.4·tanh((CRCL − 90)/40)`, written out as
/// `2·inv_logit(2u) − 1` because **`tanh` is not in the `[individual_parameters]` function
/// set and unknown unary functions there evaluate to their argument rather than erroring**
/// — so the natural spelling would silently simulate a *linear* `CRCL` effect. (That
/// silent-identity behaviour is worth fixing on its own; it is not this file's job.)
const TRUTH_SRC: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 50.0)
  theta TVV(20.0, 1.0, 500.0)
  theta WT_CL(0.75, 0.0, 3.0)
  theta WT_V(1.0, 0.0, 3.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * (WT / 70)^WT_CL * (1 + 0.4 * (2 * inv_logit(2 * (CRCL - 90) / 40) - 1)) * exp(ETA_CL)
  V  = TVV  * (WT / 70)^WT_V  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// The fitted model: the same PK structure, but the typical values come from a network.
///
/// `init` is not optional in practice — `BUSULFAN_DCM_EXPERIMENT.md` records a bare-NN VI
/// run landing at a tightness ratio of 306 (i.e. a useless bound) against 0.54 with `init`
/// declared. `center`/`scale` standardize `WT` for the same reason
/// (`docs/model-file/covariate-nn.qmd` § "Normalize your inputs").
fn dcm_src(layers: &str) -> String {
    format!(
        r"
[parameters]
  omega ETA_CL ~ 0.05
  omega ETA_V  ~ 0.05
  sigma PROP_ERR ~ 0.09 (sd)

[covariate_nn TYPICAL_PK]
  inputs     = [WT, CRCL]
  center     = [70, 95]
  scale      = [14, 30]
  outputs    = [CL, V]
  layers     = [{layers}]
  activation = tanh
  output     = softplus
  init       = [1.0, 20.0]

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V  = TYPICAL_PK.V  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"
    )
}

fn parse(src: &str) -> CompiledModel {
    ferx_core::parser::model_parser::parse_model_string(src).expect("fixture parses")
}

/// The truth, in the analytic model's parameter layout.
fn truth(model: &CompiledModel) -> ModelParameters {
    let mut p = model.default_params.clone();
    let set = |p: &mut ModelParameters, name: &str, v: f64| {
        let i = p
            .theta_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("theta {name} not found"));
        p.theta[i] = v;
    };
    set(&mut p, "TVCL", TRUE_TVCL);
    set(&mut p, "TVV", TRUE_TVV);
    set(&mut p, "WT_CL", TRUE_WT_CL);
    set(&mut p, "WT_V", TRUE_WT_V);
    p.omega = ferx_core::OmegaMatrix::from_diagonal(
        &[TRUE_OMEGA_CL, TRUE_OMEGA_V],
        vec!["ETA_CL".into(), "ETA_V".into()],
    );
    p.sigma.values = vec![TRUE_SIGMA];
    p
}

// ---------------------------------------------------------------------------
// Design + simulation
// ---------------------------------------------------------------------------

/// One 100 mg IV bolus, sampled at 0.5, 2, 6 and 12 h.
///
/// Four samples over roughly three half-lives at the typical subject, which is enough to
/// identify `CL` and `V` per subject. That matters for this test specifically: `Ω` can only
/// be recovered if each subject's own `η` is informed by its data, so a design too sparse
/// to pin `η` would produce shrinkage-driven `Ω` collapse for reasons that have nothing to
/// do with VI or with the network.
fn design(cov: CovariateDesign) -> Population {
    let subjects = (0..N_SUBJECTS)
        .map(|s| {
            let obs_times = vec![0.5, 2.0, 6.0, 12.0];
            let n_obs = obs_times.len();
            let mut covariates = HashMap::new();
            covariates.insert("WT".to_string(), cov.weight_of(s));
            covariates.insert("CRCL".to_string(), cov.crcl_of(s));
            Subject {
                id: format!("{}", s + 1),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times,
                obs_raw_times: Vec::new(),
                observations: vec![0.0; n_obs],
                obs_cmts: vec![1; n_obs],
                covariates,
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                reset_covariates: Vec::new(),
                cens: vec![0; n_obs],
                occasions: Vec::new(),
                obs_l2: Vec::new(),
                dose_occasions: Vec::new(),
                fremtype: Vec::new(),
                obs_records: vec![],
            }
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec!["WT".to_string(), "CRCL".to_string()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

/// Simulate under `truth` and write the drawn DVs back onto the design, giving a dataset
/// whose true `Ω` is known by construction.
fn simulated_population(cov: CovariateDesign) -> Population {
    let truth_model = parse(TRUTH_SRC);
    let mut pop = design(cov);
    let rows = simulate_with_seed(&truth_model, &pop, &truth(&truth_model), 1, SEED);

    let mut per_subject: HashMap<String, Vec<f64>> = HashMap::new();
    for r in &rows {
        if let SimOutcome::Continuous { value } = r.outcome {
            per_subject.entry(r.id.clone()).or_default().push(value);
        }
    }
    for subj in &mut pop.subjects {
        let dv = per_subject
            .remove(&subj.id)
            .unwrap_or_else(|| panic!("no simulated rows for subject {}", subj.id));
        assert_eq!(
            dv.len(),
            subj.observations.len(),
            "simulated row count must match the design for subject {}",
            subj.id
        );
        assert!(
            dv.iter().all(|v| v.is_finite() && *v > 0.0),
            "simulation produced a non-positive DV for subject {}",
            subj.id
        );
        subj.observations = dv;
    }
    pop
}

// ---------------------------------------------------------------------------
// Fitting
// ---------------------------------------------------------------------------

fn vi_opts() -> FitOptions {
    FitOptions {
        method: EstimationMethod::Vi,
        vi_seed: Some(SEED),
        vi_final_ofv: ViFinalOfv::Laplace,
        run_covariance_step: false,
        ..Default::default()
    }
}

fn focei_opts() -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        run_covariance_step: false,
        ..Default::default()
    }
}

fn fit_with(model: &CompiledModel, pop: &Population, opts: &FitOptions) -> FitResult {
    fit(model, pop, &model.default_params, opts).expect("fit returns")
}

fn rel_err(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs()
}

/// `Ω` diagonal as `(ω²(CL), ω²(V))`.
fn omega_diag(f: &FitResult) -> (f64, f64) {
    (f.omega[(0, 0)], f.omega[(1, 1)])
}

/// Estimated-parameter count: every `θ` (the NN's weights are ordinary thetas), plus the
/// two `Ω` diagonal entries and the one `σ`.
fn n_estimated(model: &CompiledModel) -> usize {
    model.default_params.theta.len() + 2 + 1
}

/// `AIC = −2 log L + 2k`, on the honest marginal likelihood `vi_final_ofv = laplace`
/// puts in `FitResult::ofv` — never on the ELBO, which is a bound and not comparable
/// across models (`docs/estimation/vi.qmd` § "The OFV problem").
fn aic(f: &FitResult, model: &CompiledModel) -> f64 {
    f.ofv + 2.0 * n_estimated(model) as f64
}

/// Total recovered IIV as a single number, on the log scale so that "10× too large" and
/// "10× too small" are penalised equally. A plain relative error would let a collapse
/// toward zero look like a near-miss (bounded by 1) while an explosion looks catastrophic
/// (unbounded), and both directions are failures.
fn log_miss(cl: f64, v: f64) -> f64 {
    (cl.max(1e-12) / TRUE_OMEGA_CL).ln().abs() + (v.max(1e-12) / TRUE_OMEGA_V).ln().abs()
}

fn report(tag: &str, f: &FitResult) {
    let (o_cl, o_v) = omega_diag(f);
    eprintln!(
        "{tag:>22} | w2(CL) {o_cl:.5} ({:+.1}%) | w2(V) {o_v:.5} ({:+.1}%) | \
         sigma {:.4} ({:+.1}%) | OFV {:.2}",
        (o_cl - TRUE_OMEGA_CL) / TRUE_OMEGA_CL * 100.0,
        (o_v - TRUE_OMEGA_V) / TRUE_OMEGA_V * 100.0,
        f.sigma[0],
        (f.sigma[0] - TRUE_SIGMA) / TRUE_SIGMA * 100.0,
        f.ofv,
    );
    if let Some(vi) = &f.vi {
        eprintln!(
            "{:>22} | iters {} | converged {} | tightness {:.3} | n_fd_subjects {}",
            "", vi.n_iterations, vi.converged, vi.elbo_tightness_ratio, vi.n_fd_subjects,
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Guard on the fixtures themselves — the property every other test in this file rests on.
///
/// Not gated: it fits nothing and runs in microseconds, so it registers on every PR. Its
/// job is to fail loudly if someone edits `weight_of` / `crcl_of` in a way that quietly
/// turns `Tied` into a subject-unique design (which would make Claim 4 vacuous) or flattens
/// `SubjectUnique` onto a line (which is the fixture that failed to reproduce the collapse
/// — see [`iiv_collapses_when_the_network_can_index_subjects`]).
#[test]
fn the_two_covariate_designs_are_what_they_claim_to_be() {
    let key = |x: f64| (x * 1e6).round() as i64;
    let patterns = |cov: CovariateDesign| {
        (0..N_SUBJECTS)
            .map(|i| (key(cov.weight_of(i)), key(cov.crcl_of(i))))
            .collect::<std::collections::HashSet<_>>()
            .len()
    };

    assert_eq!(
        patterns(CovariateDesign::Tied),
        N_WT_LEVELS,
        "the Tied design must present exactly {N_WT_LEVELS} distinct (WT, CRCL) pairs, so \
         the typical-value function cannot separate the 5 subjects sharing each one"
    );
    assert_eq!(
        patterns(CovariateDesign::SubjectUnique),
        N_SUBJECTS,
        "the SubjectUnique design must give every subject its own (WT, CRCL) pair"
    );

    // And the 2-D-ness of SubjectUnique: WT is monotone in `i`, so if CRCL were too, the
    // pair would be a curve in the plane and the network would face a 1-D problem. Assert
    // the two are close to uncorrelated rather than merely distinct.
    let (wt, crcl): (Vec<f64>, Vec<f64>) = (0..N_SUBJECTS)
        .map(|i| {
            (
                CovariateDesign::SubjectUnique.weight_of(i),
                CovariateDesign::SubjectUnique.crcl_of(i),
            )
        })
        .unzip();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (mw, mc) = (mean(&wt), mean(&crcl));
    let cov_wc: f64 = wt.iter().zip(&crcl).map(|(a, b)| (a - mw) * (b - mc)).sum();
    let sd = |v: &[f64], m: f64| v.iter().map(|x| (x - m).powi(2)).sum::<f64>().sqrt();
    let r = cov_wc / (sd(&wt, mw) * sd(&crcl, mc));
    assert!(
        r.abs() < 0.2,
        "WT and CRCL must be near-uncorrelated under SubjectUnique so the covariates span \
         a genuine 2-D space; got r = {r:.3}"
    );
}

/// Claim 1 — the missing VI_PLAN §6 item 13: VI recovers the between-subject variance on a
/// DCM, not just `θ` and `σ`.
///
/// This is the assertion whose absence let a DCM `Ω` shortfall go unnoticed. Run on the
/// **identified** design, so it is a statement about the estimator and not about what the
/// covariates permit; [`iiv_collapses_when_the_network_can_index_subjects`] is the
/// statement about the design.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn vi_recovers_iiv_on_a_dcm() {
    let pop = simulated_population(CovariateDesign::Tied);
    let dcm = parse(&dcm_src("4"));
    let f = fit_with(&dcm, &pop, &vi_opts());
    report("VI, layers=[4]", &f);

    let (omega_cl, omega_v) = omega_diag(&f);
    let vi = f.vi.as_ref().expect("a VI fit must report ViResult");

    // The fit has to be worth reading before its Ω means anything. A useless bound is a
    // failed fit, not a finding about Ω — `BUSULFAN_DCM_EXPERIMENT.md` records a broken
    // run at tightness 306 whose parameters looked superficially plausible.
    assert!(
        vi.elbo_tightness_ratio.is_finite() && vi.elbo_tightness_ratio < 5.0,
        "ELBO bound is too loose to read Omega off: tightness {:.2}",
        vi.elbo_tightness_ratio
    );

    // Ω_bsv, both etas, within a factor of ~2 of truth in each direction.
    //
    // Wide, and deliberately so: 60 simulated subjects, a stochastic estimator, and a
    // network that has to learn the WT relationship from a flat start. Pinning this to 3
    // s.f. would make it a tripwire for irrelevant changes. Measured at +34% on `ETA_CL`
    // and +27% on `ETA_V`, so the upper edge carries real headroom and the band's work is
    // done at the bottom — which is where the failure mode this file is about lives.
    assert!(
        omega_cl > 0.5 * TRUE_OMEGA_CL && omega_cl < 2.0 * TRUE_OMEGA_CL,
        "w2(CL): got {omega_cl:.5}, true {TRUE_OMEGA_CL} — IIV on CL not recovered"
    );
    assert!(
        omega_v > 0.5 * TRUE_OMEGA_V && omega_v < 2.0 * TRUE_OMEGA_V,
        "w2(V): got {omega_v:.5}, true {TRUE_OMEGA_V} — IIV on V not recovered"
    );

    // σ is the other half of the same accounting: variance the network has taken from Ω
    // does not have to show up in the residual, but variance it has *created* does. A σ
    // that is simultaneously right while Ω is low is the signature of absorption rather
    // than of a bad fit, so assert it rather than leaving it to the eye.
    assert!(
        rel_err(f.sigma[0], TRUE_SIGMA) < 0.35,
        "sigma: got {:.4}, true {TRUE_SIGMA}",
        f.sigma[0]
    );
}

/// Claim 2 — on an identified DCM design, **both** estimators recover `Ω`.
///
/// This was written as "VI recovers Ω at least as well as FOCEI", sourced from the Janssen
/// FVIII table in `docs/estimation/vi.qmd`, where FOCEI's `Ω` collapses onto a degenerate
/// ridge (`ω²(V₁)` ~930× truth at correlation 0.996) and VI's does not. On its first run
/// that assertion **failed**, narrowly and in the unglamorous direction: log-miss VI 0.528
/// versus FOCEI 0.423, with `ω²(CL)` at 0.1207 (VI) and 0.1119 (FOCEI) against a truth of
/// 0.09. Neither collapsed; FOCEI was simply a little closer.
///
/// So FOCEI's DCM instability is a property of the Janssen design — a multi-branch network
/// with a correlated `block_omega` and 3 observations per subject — and not a general fact
/// about FOCEI on deep compartment models. Asserting the ordering here would pin
/// optimizer-level noise on a design that does not exhibit the phenomenon, so the ordering
/// is **reported and not asserted**. What is asserted is the claim that actually holds and
/// that anyone reading a DCM fit cares about: on a design that identifies `Ω`, neither
/// estimator loses it.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn vi_and_focei_both_recover_iiv_on_an_identified_dcm() {
    let pop = simulated_population(CovariateDesign::Tied);
    let dcm = parse(&dcm_src("4"));

    let f_vi = fit_with(&dcm, &pop, &vi_opts());
    let f_focei = fit_with(&dcm, &pop, &focei_opts());
    report("VI", &f_vi);
    report("FOCEI", &f_focei);

    let (vi_cl, vi_v) = omega_diag(&f_vi);
    let (fo_cl, fo_v) = omega_diag(&f_focei);

    // Reported, not asserted — see the doc comment. Kept because the number is the whole
    // evidence base for "use VI on a DCM", and a nightly log of it is how a regression in
    // either estimator would first become visible.
    eprintln!(
        "      log-miss on Omega | VI {:.3} | FOCEI {:.3}",
        log_miss(vi_cl, vi_v),
        log_miss(fo_cl, fo_v)
    );

    for (tag, cl, v) in [("VI", vi_cl, vi_v), ("FOCEI", fo_cl, fo_v)] {
        assert!(
            cl > 0.5 * TRUE_OMEGA_CL && cl < 2.0 * TRUE_OMEGA_CL,
            "{tag} w2(CL): got {cl:.5}, true {TRUE_OMEGA_CL} — IIV on CL not recovered \
             on a design that identifies it"
        );
        assert!(
            v > 0.4 * TRUE_OMEGA_V && v < 2.5 * TRUE_OMEGA_V,
            "{tag} w2(V): got {v:.5}, true {TRUE_OMEGA_V} — IIV on V not recovered \
             on a design that identifies it"
        );
    }
}

/// Claim 3 — the mechanism, isolated: `Ω` collapses when the network has enough capacity
/// to index subjects through subject-unique covariates.
///
/// Same data, same estimator, same seed, same `init`; only the hidden width changes, on a
/// design with 60 distinct `(WT, CRCL)` pairs for 60 subjects. `layers = [4]` is 22 weights
/// and cannot separate subjects; `layers = [8, 8]` is 114 weights against 60 subjects and
/// can. This is the in-repo analogue of the busulfan fit (74 weights, 60 distinct
/// `(WT, CRCL)` pairs, `ω²(CL)` 5× low) and the test that would have caught it.
///
/// **Two inputs, not one.** An earlier version of this test used `WT` alone and did not
/// reproduce the collapse — `ω²(CL)` went 0.1256 at `[4]` to 0.1074 at `[8,8]`, *toward*
/// truth. 60 distinct values on a single axis still ask a `tanh` network for a wiggly
/// interpolant over an ordered domain, and its smoothness bias makes that a hard optimum
/// for Adam to reach. Busulfan had two subject-unique covariates; scattered in 2-D, 60
/// points separate easily. See [`CovariateDesign::crcl_of`] for why the scatter must be
/// independent of `WT`.
///
/// It is a **characterization** test: it asserts that the collapse happens today, because
/// it does and because nothing else in the repo records that it does. When
/// `NN_REGULARIZATION_PLAN.md` lands, `nn_l2` should shrink or remove this gap — at which
/// point this test fails and is the natural home for the before/after number. That is the
/// intended lifecycle, not an accident: a silent pass after the fix would prove nothing.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn iiv_collapses_when_the_network_can_index_subjects() {
    let pop = simulated_population(CovariateDesign::SubjectUnique);

    let m_narrow = parse(&dcm_src("4"));
    let m_wide = parse(&dcm_src("8, 8"));
    let f_narrow = fit_with(&m_narrow, &pop, &vi_opts());
    let f_wide = fit_with(&m_wide, &pop, &vi_opts());
    report("unique cov, [4]", &f_narrow);
    report("unique cov, [8,8]", &f_wide);

    let (narrow_cl, _) = omega_diag(&f_narrow);
    let (wide_cl, _) = omega_diag(&f_wide);
    eprintln!(
        "      w2(CL) [4] {narrow_cl:.5} ({} weights) -> [8,8] {wide_cl:.5} ({} weights), \
         true {TRUE_OMEGA_CL}",
        n_estimated(&m_narrow) - 3,
        n_estimated(&m_wide) - 3,
    );

    // The capacity the design can support recovers Ω, so the collapse below is
    // attributable to capacity and not to the design being unidentifiable outright.
    assert!(
        narrow_cl > 0.5 * TRUE_OMEGA_CL,
        "w2(CL) {narrow_cl:.5} at layers=[4] (true {TRUE_OMEGA_CL}): the narrow network \
         should still recover IIV even with subject-unique covariates, or this design is \
         broken for reasons other than capacity"
    );

    // The collapse itself. The busulfan fit landed at 0.19× truth; half of truth is the
    // line between "biased low" (mechanism 1, mild, documented) and "the network took it"
    // (mechanism 2).
    assert!(
        wide_cl < 0.5 * TRUE_OMEGA_CL,
        "w2(CL) {wide_cl:.5} at layers=[8,8] (true {TRUE_OMEGA_CL}): expected the network \
         to absorb the between-subject variation on a subject-unique covariate. If this \
         now passes because NN regularization landed, that is the fix working — update \
         this test to assert recovery and record the before/after."
    );

    // And the actionable half: standard model selection already rejects the collapsed fit,
    // without being told anything about Ω. Computed on `vi_final_ofv = laplace`'s honest
    // marginal likelihood, never on the ELBO.
    //
    // Note this is a conservative guard rather than a detector: AIC penalizes capacity
    // unconditionally, so it also prefers the narrow net on designs where the wide one is
    // more accurate. It points the right way here, which is what a user needs.
    let (aic_narrow, aic_wide) = (aic(&f_narrow, &m_narrow), aic(&f_wide, &m_wide));
    eprintln!("      AIC | [4] {aic_narrow:.2} | [8,8] {aic_wide:.2}");
    assert!(
        aic_wide > aic_narrow,
        "AIC should reject the over-capacity network that collapsed Omega: \
         [8,8] {aic_wide:.2} vs [4] {aic_narrow:.2}"
    );
}

/// Claim 4 — the contrast that makes Claim 3 a statement about the *design*.
///
/// The same `[8, 8]` network that collapses `Ω` on subject-unique weights does not collapse
/// it when the weights are tied into 12 levels. Capacity alone is therefore not the
/// variable; capacity **relative to the number of distinct covariate patterns** is.
///
/// This is what the first version of this file got backwards. It ran the capacity sweep on
/// the tied design only, asserted `wide < narrow`, and passed — but measured `ω²(CL)` going
/// 0.1207 → 0.0921 against a truth of 0.09, i.e. the wide network was *more* accurate and
/// the "absorption" it detected was the narrow network's bias being corrected.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn capacity_alone_does_not_collapse_iiv_when_the_design_identifies_omega() {
    let pop = simulated_population(CovariateDesign::Tied);
    let f = fit_with(&parse(&dcm_src("8, 8")), &pop, &vi_opts());
    report("tied cov, [8,8]", &f);

    let (omega_cl, omega_v) = omega_diag(&f);
    assert!(
        omega_cl > 0.5 * TRUE_OMEGA_CL && omega_cl < 2.0 * TRUE_OMEGA_CL,
        "w2(CL) {omega_cl:.5} (true {TRUE_OMEGA_CL}): a wide network must not absorb IIV \
         when tied covariate levels leave it nothing to index subjects with"
    );
    assert!(
        omega_v > 0.4 * TRUE_OMEGA_V && omega_v < 2.5 * TRUE_OMEGA_V,
        "w2(V) {omega_v:.5}, true {TRUE_OMEGA_V}"
    );
}
