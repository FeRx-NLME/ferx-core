//! #2973: a SAEM fit reports the importance-sampled −2 log L (NONMEM `IMP EONLY=1`)
//! as its OFV, with the FOCE/Laplace value kept as a labelled diagnostic.
//!
//! Tier 2: every fit here is a handful of SAEM iterations, and every anchor is one
//! importance-sampling evaluation at fixed estimates.
//!
//! The NONMEM anchors: `nonmem_anchor/results/*.ext` of the SAEM control streams that end in
//! `METHOD=IMP EONLY=1 NITER=10 ISAMPLE=3000`. NONMEM's SAEM table holds its final estimates
//! (table 1); its IMP table holds the ten evaluation iterations (table 2), each an independent
//! K = 3000 draw whose spread is NONMEM's own Monte-Carlo error. ferx evaluates its importance
//! sampler at **NONMEM's** estimates, so the two objectives are the same integral at the same
//! point, and must agree within their combined Monte-Carlo errors.

use ferx_core::types::{FitOptions, FitResult, ModelParameters, OmegaMatrix, SaemOfv};
use ferx_core::{fit, parse_model_file, prepare_run, read_nonmem_csv, EstimationMethod};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn short_saem(ofv: SaemOfv) -> FitOptions {
    FitOptions {
        method: EstimationMethod::Saem,
        saem_n_exploration: 10,
        saem_n_convergence: 10,
        saem_seed: Some(11),
        imp_seed: Some(5),
        saem_ofv: ofv,
        saem_ofv_samples: 500,
        run_covariance_step: false,
        threads: Some(1),
        verbose: false,
        ..FitOptions::default()
    }
}

fn warfarin_fit(opts: &FitOptions) -> FitResult {
    let model = parse_model_file(Path::new("examples/warfarin.ferx")).expect("model");
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("data");
    fit(&model, &pop, &model.default_params, opts).expect("fit")
}

// ---------------------------------------------------------------------------
// Option parse / route
// ---------------------------------------------------------------------------

#[test]
fn saem_ofv_defaults_to_imp_and_parses_each_spelling() {
    let d = FitOptions::default();
    assert_eq!(d.saem_ofv, SaemOfv::Imp);
    assert_eq!(d.saem_ofv_samples, 3000);
    assert_eq!(d.saem_ofv_nodes, 11);

    let src = std::fs::read_to_string(root().join("examples/warfarin.ferx")).unwrap();
    let with = |line: &str| {
        let head = src.split("[fit_options]").next().unwrap();
        let text = format!("{head}\n[fit_options]\n  method = saem\n  {line}\n");
        ferx_core::parser::model_parser::parse_full_model(&text)
    };
    for (v, want) in [
        ("saem_ofv = imp", SaemOfv::Imp),
        ("saem_ofv = agq", SaemOfv::Agq),
        ("saem_ofv = focei", SaemOfv::Focei),
    ] {
        let parsed = with(v).unwrap_or_else(|e| panic!("{v}: {e}"));
        assert_eq!(parsed.fit_options.saem_ofv, want, "{v}");
    }
    let parsed = with("saem_ofv_samples = 1234").unwrap();
    assert_eq!(parsed.fit_options.saem_ofv_samples, 1234);
    let parsed = with("saem_ofv_pilot_samples = 123").unwrap();
    assert_eq!(parsed.fit_options.saem_ofv_pilot_samples, 123);
    let parsed = with("imp_adapt_samples = 77").unwrap();
    assert_eq!(parsed.fit_options.imp_adapt_samples, 77);
    let parsed = with("saem_ofv_nodes = 13").unwrap();
    assert_eq!(parsed.fit_options.saem_ofv_nodes, 13);

    let err = with("saem_ofv = bogus")
        .err()
        .expect("bogus value is rejected");
    assert!(err.contains("saem_ofv must be"), "{err}");
    assert!(with("saem_ofv_samples = 1").is_err());
    assert!(with("saem_ofv_nodes = 1").is_err());
    assert!(with("saem_ofv_nodes = 99").is_err());
}

// ---------------------------------------------------------------------------
// The reported objective
// ---------------------------------------------------------------------------

/// Default: the OFV *is* the importance-sampled value, its SE is published, AIC/BIC are built
/// from it, and the FOCE/Laplace value survives on the side — equal to what `saem_ofv = focei`
/// reports for the same (seeded) SAEM run.
#[test]
fn default_saem_fit_reports_the_is_objective_with_the_foce_value_beside_it() {
    let imp = warfarin_fit(&short_saem(SaemOfv::Imp));
    let focei = warfarin_fit(&short_saem(SaemOfv::Focei));

    // The same seeded SAEM run: the estimates, hence the FOCE value, do not depend on the
    // readout.
    assert_eq!(
        imp.theta, focei.theta,
        "the readout must not move estimates"
    );

    let obj = imp
        .saem_objective
        .as_ref()
        .expect("default reports a SAEM objective");
    let is = imp
        .importance_sampling
        .as_ref()
        .expect("IS result is published");
    assert_eq!(obj.kind, SaemOfv::Imp);
    assert_eq!(imp.ofv.to_bits(), is.minus2_log_likelihood.to_bits());
    assert_eq!(obj.mc_standard_error, Some(is.mc_standard_error));
    assert_eq!(
        obj.n_samples,
        Some(500),
        "saem_ofv_samples, not imp_samples"
    );
    assert_eq!(is.n_samples, 500);
    assert!(is.mc_standard_error > 0.0 && is.mc_standard_error.is_finite());

    assert!(
        focei.saem_objective.is_none(),
        "saem_ofv = focei keeps today's behaviour"
    );
    assert!(focei.importance_sampling.is_none());
    assert_eq!(obj.foce_ofv.to_bits(), focei.ofv.to_bits());
    assert!(
        (imp.ofv - obj.foce_ofv).abs() > 1e-6,
        "the IS value and the Laplace value are different numbers"
    );

    // AIC / BIC come from the reported (IS) objective.
    let k = imp.n_parameters as f64;
    assert!((imp.aic - (imp.ofv + 2.0 * k)).abs() < 1e-9);
    assert!((focei.aic - (focei.ofv + 2.0 * k)).abs() < 1e-9);
    assert!((imp.aic - focei.aic - (imp.ofv - focei.ofv)).abs() < 1e-9);
}

/// Seeded and deterministic: two default fits agree to the bit.
#[test]
fn default_saem_objective_is_deterministic() {
    let a = warfarin_fit(&short_saem(SaemOfv::Imp));
    let b = warfarin_fit(&short_saem(SaemOfv::Imp));
    assert_eq!(a.ofv.to_bits(), b.ofv.to_bits());
}

/// `agq`: deterministic, labelled, no Monte-Carlo SE, no IS result; its value agrees with the
/// importance-sampled one within the IS standard error, and is converged in the node count
/// (11 vs 15 nodes; the bench fit that chose 11 is in `saem_ofv_nodes`).
#[test]
fn agq_readout_is_deterministic_and_agrees_with_is_within_its_se() {
    let mut o = short_saem(SaemOfv::Agq);
    let a = warfarin_fit(&o);
    let b = warfarin_fit(&o);
    assert_eq!(a.ofv.to_bits(), b.ofv.to_bits());
    let obj = a.saem_objective.as_ref().expect("agq objective");
    assert_eq!(obj.kind, SaemOfv::Agq);
    assert_eq!(obj.n_nodes, Some(11));
    assert!(obj.mc_standard_error.is_none() && obj.n_samples.is_none());
    assert!(a.importance_sampling.is_none());

    o.saem_ofv_nodes = 15;
    let eleven = warfarin_fit(&o);
    assert_eq!(a.theta, eleven.theta);
    let node_gap = (a.ofv - eleven.ofv).abs();

    let mut oi = short_saem(SaemOfv::Imp);
    oi.saem_ofv_samples = 20_000;
    let imp = warfarin_fit(&oi);
    let se = imp
        .saem_objective
        .as_ref()
        .unwrap()
        .mc_standard_error
        .unwrap();
    let gap = (eleven.ofv - imp.ofv).abs();
    eprintln!(
        "agq11 {:.5}  agq15 {:.5}  (node gap {node_gap:.2e})  imp(K=20000) {:.5} (SE {se:.4})  \
         |agq15-imp| {gap:.4}",
        a.ofv, eleven.ofv, imp.ofv
    );
    assert!(node_gap < 0.05, "AGQ is not converged in nodes: {node_gap}");
    assert!(
        gap < 3.0 * se,
        "AGQ {} vs IS {} (SE {se})",
        eleven.ofv,
        imp.ofv
    );
    assert!((a.foce_diag() - imp.saem_objective.as_ref().unwrap().foce_ofv).abs() < 1e-9);
}

trait FoceDiag {
    fn foce_diag(&self) -> f64;
}
impl FoceDiag for FitResult {
    fn foce_diag(&self) -> f64 {
        self.saem_objective.as_ref().unwrap().foce_ofv
    }
}

/// A model the agq readout cannot grid (more than three random effects) falls back to `imp`
/// with a warning that names the reason, instead of failing the fit.
#[test]
fn agq_falls_back_to_imp_beyond_three_random_effects() {
    let src = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVKA(1.0, 0.1, 10.0)
  theta TVQ(5.0, 0.1, 100.0)
  theta TVV2(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.1
  omega ETA_V ~ 0.1
  omega ETA_KA ~ 0.1
  omega ETA_Q ~ 0.1
  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  Q = TVQ * exp(ETA_Q)
  V2 = TVV2

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = ferx_core::parse_model_string(src).expect("model");
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("data");
    let r = fit(
        &model,
        &pop,
        &model.default_params,
        &short_saem(SaemOfv::Agq),
    )
    .expect("fit");
    let obj = r
        .saem_objective
        .as_ref()
        .expect("falls back to a reported objective");
    assert_eq!(obj.kind, SaemOfv::Imp);
    assert!(
        r.warnings
            .iter()
            .any(|w| w.contains("saem_ofv = agq unavailable") && w.contains("three random effects")),
        "{:?}",
        r.warnings
    );
}

/// `[saem, imp]` + `imp_eval_only` is the explicit spelling of the same workflow: its IS value is
/// the reported OFV under the default, with no second readout (the chain's IS result is the one
/// published, at `imp_samples`, not `saem_ofv_samples`); under `saem_ofv = focei` it leaves the
/// FOCE/Laplace `ofv` alone, as before. A trailing `agq_eval_only` stage is its own explicit
/// readout and gets no `saem_objective`.
#[test]
fn explicit_evaluator_chains_follow_saem_ofv() {
    let chained = |ofv: SaemOfv| {
        let mut o = short_saem(ofv);
        o.methods = vec![EstimationMethod::Saem, EstimationMethod::Imp];
        o.imp_eval_only = true;
        o.imp_samples = 400;
        warfarin_fit(&o)
    };
    let imp = chained(SaemOfv::Imp);
    let obj = imp
        .saem_objective
        .as_ref()
        .expect("explicit imp stage is the reported objective");
    let is = imp.importance_sampling.as_ref().unwrap();
    assert_eq!(
        is.n_samples, 400,
        "the chain's own imp_samples, not saem_ofv_samples"
    );
    assert_eq!(obj.n_samples, Some(400));
    assert_eq!(imp.ofv.to_bits(), is.minus2_log_likelihood.to_bits());

    let focei = chained(SaemOfv::Focei);
    assert!(focei.saem_objective.is_none());
    assert_eq!(
        focei.ofv.to_bits(),
        obj.foce_ofv.to_bits(),
        "ofv stays FOCE/Laplace"
    );
    assert!(
        focei.importance_sampling.is_some(),
        "the explicit stage still publishes its IS result"
    );

    let mut o = short_saem(SaemOfv::Imp);
    o.methods = vec![EstimationMethod::Saem, EstimationMethod::Laplace];
    o.agq_eval_only = true;
    o.n_agq = 7;
    let agq_stage = warfarin_fit(&o);
    assert!(agq_stage.saem_objective.is_none());
    assert!(
        agq_stage.importance_sampling.is_none(),
        "no second readout on an explicit chain"
    );
}

/// The pilot passes that fit the proposal draw `imp_adapt_samples` (`saem_ofv_pilot_samples` for
/// the SAEM readout); the final readout always draws K. `0` and any count at or above K are the
/// full-K pilots, bit for bit, a smaller count changes the proposal and so the draws, and the
/// reported K is the readout's.
#[test]
fn pilot_samples_shape_the_proposal_and_never_the_readout_size() {
    let model = parse_model_file(Path::new("examples/warfarin.ferx")).expect("model");
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("data");
    let eval = |pilot: usize| {
        let o = FitOptions {
            method: EstimationMethod::Imp,
            methods: vec![],
            imp_eval_only: true,
            imp_samples: 400,
            imp_adapt_samples: pilot,
            imp_seed: Some(3),
            outer_maxiter: 0,
            run_covariance_step: false,
            threads: Some(1),
            verbose: false,
            ..FitOptions::default()
        };
        fit(&model, &pop, &model.default_params, &o)
            .expect("IS evaluation")
            .importance_sampling
            .expect("marginal")
    };
    let full = eval(0);
    for same in [eval(400), eval(10_000)] {
        assert_eq!(
            same.minus2_log_likelihood.to_bits(),
            full.minus2_log_likelihood.to_bits()
        );
    }
    let small = eval(60);
    assert_ne!(
        small.minus2_log_likelihood.to_bits(),
        full.minus2_log_likelihood.to_bits()
    );
    assert_eq!(small.n_samples, 400, "the readout still draws imp_samples");
    // Both are unbiased estimates of one integral: within a few combined SEs.
    let se = (small.mc_standard_error.powi(2) + full.mc_standard_error.powi(2)).sqrt();
    assert!(
        (small.minus2_log_likelihood - full.minus2_log_likelihood).abs() < 4.0 * se,
        "{} vs {} (SE {se})",
        small.minus2_log_likelihood,
        full.minus2_log_likelihood
    );
}

/// The SAEM readout hands `saem_ofv_pilot_samples` to the evaluation: a pilot of K is the
/// full-K readout bit for bit, and a smaller one is a different (equally valid) draw.
#[test]
fn saem_readout_reads_saem_ofv_pilot_samples() {
    let run = |pilot: usize| {
        let mut o = short_saem(SaemOfv::Imp);
        o.saem_ofv_pilot_samples = pilot;
        warfarin_fit(&o)
    };
    let full = run(0);
    assert_eq!(
        run(500).ofv.to_bits(),
        full.ofv.to_bits(),
        "a pilot of K is the full-K pilot"
    );
    let small = run(80);
    assert_ne!(
        small.ofv.to_bits(),
        full.ofv.to_bits(),
        "the option reaches the evaluation"
    );
    assert_eq!(small.saem_objective.as_ref().unwrap().n_samples, Some(500));
    assert_eq!(FitOptions::default().saem_ofv_pilot_samples, 300);
}

/// The labels: console report and YAML say what the OFV is and carry the diagnostic.
#[test]
fn outputs_label_the_objective_and_carry_the_foce_diagnostic() {
    let dir = std::env::temp_dir().join(format!("ferx_2973_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let imp = warfarin_fit(&short_saem(SaemOfv::Imp));
    let yaml_path = dir.join("fit.yaml");
    ferx_core::io::output::write_estimates_yaml(&imp, yaml_path.to_str().unwrap()).expect("yaml");
    let yaml = std::fs::read_to_string(&yaml_path).unwrap();
    assert!(yaml.contains("  ofv_type: importance_sampling\n"), "{yaml}");
    assert!(yaml.contains("  ofv_mc_standard_error: "), "{yaml}");
    assert!(yaml.contains("  ofv_n_samples: 500\n"), "{yaml}");
    assert!(yaml.contains("  ofv_foce_laplace: "), "{yaml}");

    let focei = warfarin_fit(&short_saem(SaemOfv::Focei));
    ferx_core::io::output::write_estimates_yaml(&focei, yaml_path.to_str().unwrap()).expect("yaml");
    let yaml = std::fs::read_to_string(&yaml_path).unwrap();
    assert!(
        !yaml.contains("ofv_type:") && !yaml.contains("ofv_foce_laplace"),
        "{yaml}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// NONMEM anchors
// ---------------------------------------------------------------------------

struct ExtTable {
    /// The `ITERATION = -1000000000` row, empty when the run ended before writing it.
    last: HashMap<String, f64>,
    /// The final iteration row, which is what NONMEM reports when the `-1000000000` row is
    /// missing (the `logit_fraction_saem` stream: its `.lst` aborts after both stages).
    last_iter: HashMap<String, f64>,
    /// `OBJ` of each iteration row (`ITERATION >= 0`).
    iter_obj: Vec<f64>,
}

fn ext_tables(name: &str) -> Vec<ExtTable> {
    let text = std::fs::read_to_string(root().join("nonmem_anchor/results").join(name)).unwrap();
    let mut tables: Vec<ExtTable> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first() {
            None => continue,
            Some(&"TABLE") => tables.push(ExtTable {
                last: HashMap::new(),
                last_iter: HashMap::new(),
                iter_obj: vec![],
            }),
            Some(&"ITERATION") => names = f.iter().map(|s| s.to_string()).collect(),
            Some(first) => {
                let Ok(it) = first.parse::<i64>() else {
                    continue;
                };
                let vals: HashMap<String, f64> = names
                    .iter()
                    .cloned()
                    .zip(f.iter().map(|v| v.parse::<f64>().unwrap()))
                    .collect();
                let t = tables.last_mut().expect("row before TABLE");
                if it == -1_000_000_000 {
                    t.last = vals;
                } else if it >= 0 {
                    if let Some(o) = vals.get("OBJ") {
                        t.iter_obj.push(*o);
                    }
                    t.last_iter = vals;
                }
            }
        }
    }
    for t in tables.iter_mut().filter(|t| t.last.is_empty()) {
        t.last = t.last_iter.clone();
    }
    tables
}

fn sd(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}

/// ferx's importance-sampled −2 log L at NONMEM's SAEM estimates.
fn ferx_is_at(
    model: &str,
    data: &str,
    est: &HashMap<String, f64>,
    k: usize,
    pilot: usize,
) -> (f64, f64) {
    let prep = prepare_run(
        root().join("nonmem_anchor").join(model).to_str().unwrap(),
        Some(root().join(data).to_str().unwrap()),
    )
    .expect("prepare");
    let mut p: ModelParameters = prep.init_params.clone();
    for (i, t) in p.theta.iter_mut().enumerate() {
        *t = est[&format!("THETA{}", i + 1)];
    }
    let n = p.omega.matrix.nrows();
    let var: Vec<f64> = (1..=n).map(|i| est[&format!("OMEGA({i},{i})")]).collect();
    p.omega = OmegaMatrix::from_diagonal(&var, p.omega.eta_names.clone());
    p.sigma.values = vec![est["SIGMA(1,1)"].sqrt()];
    let opts = FitOptions {
        method: EstimationMethod::Imp,
        methods: vec![],
        interaction: true,
        imp_eval_only: true,
        imp_samples: k,
        imp_adapt_samples: pilot,
        imp_seed: Some(2973),
        outer_maxiter: 0,
        run_covariance_step: false,
        threads: Some(2),
        verbose: false,
        ..FitOptions::default()
    };
    let r = fit(&prep.parsed.model, &prep.population, &p, &opts).expect("IS evaluation");
    let is = r.importance_sampling.expect("marginal");
    (is.minus2_log_likelihood, is.mc_standard_error)
}

/// (ferx model, data, NONMEM .ext)
const ANCHORS: &[(&str, &str, &str)] = &[
    (
        "covmuref_shared_saem_fit.ferx",
        "nonmem_anchor/covmuref_shared.csv",
        "covmuref_shared_saem.ext",
    ),
    (
        "covmuref_additive_saem_fit.ferx",
        "data/covmuref_additive.csv",
        "covmuref_additive_saem.ext",
    ),
    (
        "covmuref_power_saem_fit.ferx",
        "data/covmuref_power.csv",
        "covmuref_power_saem.ext",
    ),
    (
        "covmuref_anchor_a_saem_fit.ferx",
        "nonmem_anchor/covmuref_shared.csv",
        "covmuref_anchor_a_saem.ext",
    ),
    (
        "covmuref_anchor_b_saem_fit.ferx",
        "nonmem_anchor/covmuref_shared.csv",
        "covmuref_anchor_b_saem.ext",
    ),
    (
        "logit_fraction_saem_fit.ferx",
        "data/logit_fraction_oral.csv",
        "logit_fraction_saem.ext",
    ),
];

/// ferx's IS objective at NONMEM's SAEM estimates against NONMEM's `IMP EONLY=1` objective,
/// within the combined Monte-Carlo errors. NONMEM's SAEM-table `OBJ` (its stochastic-approximation
/// quantity, −2210 on the shared-θ anchor against an exact −1199) is what the old FOCE/Laplace
/// readout stood in for: the gap between it and the EONLY value is the reason for the change.
#[test]
fn is_objective_matches_nonmem_eonly_at_nonmem_estimates() {
    for (model, data, ext) in ANCHORS {
        let tables = ext_tables(ext);
        assert_eq!(tables.len(), 2, "{ext}: SAEM table + IMP EONLY table");
        let (saem, imp) = (&tables[0], &tables[1]);
        let nm_obj = imp.last["OBJ"];
        let nm_se = sd(&imp.iter_obj[1..]);
        // Full-size pilots (NONMEM's own K per iteration) and the SAEM readout's default pilots.
        for pilot in [0, FitOptions::default().saem_ofv_pilot_samples] {
            let (ferx, ferx_se) = ferx_is_at(model, data, &saem.last, 3000, pilot);
            let z = (ferx - nm_obj) / (nm_se * nm_se + ferx_se * ferx_se).sqrt();
            eprintln!(
                "ANCHOR {ext} pilot {pilot}: NONMEM SAEM-OBJ {:.3}  NONMEM EONLY {nm_obj:.3} \
                 (SE {nm_se:.3})  ferx IS {ferx:.3} (SE {ferx_se:.3})  diff {:+.3}  z {z:+.2}",
                saem.last["SAEMOBJ"],
                ferx - nm_obj
            );
            // Measured on Linux (debug): |z| <= 1.11 at full-size pilots and <= 1.34 with
            // 300-draw pilots, worst |diff| 0.32 OFV units; 3 is the usual 3-sigma bound on
            // a standardised difference of two independent Monte-Carlo estimates.
            assert!(
                z.abs() < 3.0,
                "{ext} (pilot {pilot}): ferx {ferx} vs NONMEM {nm_obj}, z = {z}"
            );
        }
    }
}
