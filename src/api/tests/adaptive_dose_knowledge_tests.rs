//! Which subjects an `[adaptive_dosing]` dose expression may read individual values for
//! (#944 PR2): a subject whose η are supplied may read its individual parameters and
//! random effects; a drawn subject is refused (`E_ADAPTIVE_DOSE_KNOWLEDGE`) unless the
//! block opts into `dose_knowledge = oracle`, which flags it (`dose_oracle`,
//! `W_ADAPTIVE_DOSE_ORACLE`).
//!
//! Adaptive-dosing exception (AGENTS.md): no NONMEM. The oracles are closed forms
//! computed outside the engine — `400 · CL` from the model's own formula, and, under IOV,
//! the CL in force in the first occasion backed out of the next decision's trough.
//!
//! Each test names the regression it exists to catch.

use super::*;
use crate::diagnostics::EngineError;
use crate::parser::model_parser::parse_full_model;
use crate::sim::adaptive::AdaptiveDosingSpec;
use crate::types::ParsedModel;
use std::io::Write;

const TVCL: f64 = 6.0;
const TVV: f64 = 0.7;

/// 1-cpt bolus, CL and V scaled by weight, η on CL (variance 0.09, so a drawn η is
/// visibly not 0). `kappa` adds an IOV κ on CL.
fn model(kappa: bool, block: &str) -> ParsedModel {
    model_with(kappa, "0.09", block)
}

/// [`model`] with the η variance `omega` (`0` draws η = 0 exactly, #1605).
fn model_with(kappa: bool, omega: &str, block: &str) -> ParsedModel {
    let (k_decl, k_use) = if kappa {
        ("  kappa KAPPA_CL ~ 0.5\n", " + KAPPA_CL")
    } else {
        ("", "")
    };
    parse_full_model(&format!(
        "[parameters]
  theta TVCL({TVCL}, 0.1, 50.0)
  theta TVV({TVV}, 0.05, 5.0)
  omega ETA_CL ~ {omega}
{k_decl}  sigma ADD ~ 1.0
[individual_parameters]
  CL = TVCL * (WT / 70) * exp(ETA_CL{k_use})
  V  = TVV * WT
[structural_model]
  ode(states=[CENT])
[odes]
  d/dt(CENT) = -(CL / V) * CENT
[scaling]
  y = CENT / V
[error_model]
  DV ~ additive(ADD)
[adaptive_dosing]
{block}"
    ))
    .expect("model parses")
}

/// `start_dose = <start>` with literal bounds, bolus, decisions at 0, 24, 48, and a rule
/// that never fires, so the first dose is the plan's start dose.
fn block(start: &str, extra: &str) -> String {
    format!(
        "  observe = CENT / V
  at = [0, 24, 48]
  start_dose = {start}
  route = bolus(cmt=1)
  dose_bounds = [0, 4000]
{extra}  when signal > 1000000 : decrease 25%"
    )
}

fn pop(csv: &str) -> Population {
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    f.write_all(csv.as_bytes()).expect("write csv");
    crate::read_nonmem_csv(f.path(), None, None).expect("data reads")
}

/// One subject per `(id, WT)`, observations at 0, 24 and 48.
fn csv(subjects: &[(&str, f64)]) -> String {
    let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,MDV,WT\n");
    for (id, wt) in subjects {
        for t in [0.0, 24.0, 48.0] {
            s.push_str(&format!("{id},{t},0,0,0,1,0,{wt}\n"));
        }
    }
    s
}

/// Run with η supplied for `supplied` (`(id, η_CL)`) and every other subject drawn.
fn run(
    p: &ParsedModel,
    population: &Population,
    supplied: &[(&str, f64)],
    n_sim: usize,
) -> Result<AdaptiveSimulationResult, EngineError> {
    let spec = p.adaptive_dosing.as_ref().expect("block present");
    let opts = AdaptiveSimulateOptions {
        seed: Some(17),
        ..Default::default()
    };
    let etas = crate::SuppliedEtas::from_table(
        supplied.iter().map(|(id, _)| id.to_string()).collect(),
        p.model.eta_names.clone(),
        supplied.iter().map(|&(_, e)| vec![e]).collect(),
    )
    .expect("η table");
    let inputs = crate::AdaptiveInputs::new(&p.model.default_params)
        .with_etas(&etas)
        .with_unmatched(crate::UnmatchedSubjects::Draw);
    simulate_adaptive_from_spec_with_inputs(&p.model, population, &inputs, n_sim, spec, &opts)
}

/// Run with every subject drawn, through the entry point that takes no supplied η (a
/// `[mixture]` model refuses a supplied-η table, even an empty one).
fn run_drawn(
    p: &ParsedModel,
    population: &Population,
    n_sim: usize,
) -> Result<AdaptiveSimulationResult, EngineError> {
    let spec = p.adaptive_dosing.as_ref().expect("block present");
    let opts = AdaptiveSimulateOptions {
        seed: Some(17),
        ..Default::default()
    };
    simulate_adaptive_from_spec(
        &p.model,
        population,
        &p.model.default_params,
        n_sim,
        spec,
        &opts,
    )
}

fn first_dose(r: &AdaptiveSimulationResult, id: &str, sim: usize) -> f64 {
    r.ledger
        .iter()
        .find(|e| e.subject == id && e.sim == sim && e.time == 0.0)
        .map(|e| e.amt)
        .expect("a first dose for the subject")
}

/// `400 · CL` from the model's formula, outside the engine, at κ = 0.
fn dose_400_cl(wt: f64, eta: f64) -> f64 {
    400.0 * (TVCL * (wt / 70.0) * eta.exp())
}

/// The refusal for subject 2 drawn under `known`, as a whole, so deleting any sentence
/// fails the test that asserts it.
const K1_CL: &str = "[adaptive_dosing] start_dose reads CL, an individual parameter, but \
    the random effects of 1 of 2 subject(s) are drawn by the simulation ('2'), so CL is the \
    simulated truth, which no prescriber knows at dosing time. To dose from it: carry the \
    known value as a data column (e.g. CL_KNOWN) and read that; supply each subject's \
    random effects (e.g. a previous fit's estimates, as `etas` in R, or through \
    `AdaptiveInputs::with_etas` and `simulate_adaptive_from_spec_with_inputs` in Rust); or \
    set dose_knowledge = oracle for a perfect-information comparison arm, which is flagged \
    in the results.";

// ── The straddle: supplied may read, drawn may not ────────────────────────────────

/// One expression, two runs that differ only in whether subject 2's η are supplied. With
/// subject 1 supplied and 2 drawn (`UnmatchedSubjects::Draw`), the run is refused naming
/// subject 2 **and not** subject 1. With both supplied it runs, and each first dose is
/// `400 · CL(supplied η)` from the model's formula, to the bit (measured 0 gap: the
/// expression and the formula evaluate in the same order). A random-effect read is
/// refused with its own kind.
///
/// Mutations: the knowledge check deleted (the mixed run doses subject 2 from its draw);
/// the rule decided per run (either the mixed run passes, or the all-supplied run is
/// refused); a supplied subject evaluated at its drawn η or at η = 0 (the dose is off
/// 400·CL(η)); the drawn list including supplied subjects (`'1'` named); the kind
/// hard-coded (`ETA_CL` called an individual parameter).
#[test]
fn a_supplied_subject_may_read_cl_and_a_drawn_one_is_refused() {
    let p = model(false, &block("400 * CL", ""));
    let population = pop(&csv(&[("1", 70.0), ("2", 50.0)]));

    let e = run(&p, &population, &[("1", 0.2)], 1).expect_err("subject 2 is drawn");
    assert_eq!(e.code(), Some("E_ADAPTIVE_DOSE_KNOWLEDGE"), "{e}");
    assert_eq!(e.message(), K1_CL);
    assert!(
        !e.message().contains("'1'"),
        "a supplied subject is not named"
    );

    let r = run(&p, &population, &[("1", 0.2), ("2", -0.1)], 1).expect("both supplied");
    for (id, wt, eta) in [("1", 70.0, 0.2), ("2", 50.0, -0.1)] {
        assert_eq!(
            first_dose(&r, id, 1).to_bits(),
            dose_400_cl(wt, eta).to_bits(),
            "subject {id}: {} vs {}",
            first_dose(&r, id, 1),
            dose_400_cl(wt, eta)
        );
    }
    assert!(r.metrics.iter().all(|m| m.eta_supplied && !m.dose_oracle));

    let p = model(false, &block("1000 * exp(ETA_CL)", ""));
    let e = run(&p, &population, &[("1", 0.2)], 1).expect_err("subject 2 is drawn");
    assert_eq!(e.code(), Some("E_ADAPTIVE_DOSE_KNOWLEDGE"), "{e}");
    assert!(
        e.message().starts_with(
            "[adaptive_dosing] start_dose reads ETA_CL, a random effect, but the random \
             effects of 1 of 2 subject(s) are drawn"
        ),
        "{e}"
    );
    assert!(!e.message().contains("individual parameter"), "{e}");
}

// ── The oracle, in a mixed run ─────────────────────────────────────────────────────

/// `dose_knowledge = oracle` on a run that mixes a supplied subject (1) and a drawn one
/// (2), over two replicates. Subject 1 reads its supplied η in every replicate and is not
/// flagged; subject 2 reads its draw, so its first dose differs between the replicates and
/// from `400 · CL(η = 0)`, and it is flagged in every replicate. The warning is asserted
/// whole and counts subject 2 once. With every subject supplied, nothing reads a draw, so
/// no subject is flagged and no warning is raised.
///
/// Mutations: the flag set per run (subject 1 flagged); the flag dropped; the warning
/// dropped, or counting replicates instead of subjects ("2 of 2"); a sentence of the
/// warning deleted; an oracle subject evaluated at η = 0 (equal doses across replicates);
/// a supplied subject in an oracle run evaluated at its draw; the warning raised with no
/// subject dosed from a draw.
#[test]
fn an_oracle_run_flags_each_drawn_subject_and_warns_once() {
    let p = model(false, &block("400 * CL", "  dose_knowledge = oracle\n"));
    let population = pop(&csv(&[("1", 70.0), ("2", 50.0)]));
    let r = run(&p, &population, &[("1", 0.2)], 2).expect("oracle runs");

    assert_eq!(r.metrics.len(), 4);
    for m in &r.metrics {
        assert_eq!(
            m.dose_oracle,
            m.subject == "2",
            "{} sim {}",
            m.subject,
            m.sim
        );
        assert_eq!(
            m.eta_supplied,
            m.subject == "1",
            "{} sim {}",
            m.subject,
            m.sim
        );
    }
    let w1: Vec<&String> = r
        .warnings
        .iter()
        .filter(|w| w.starts_with("W_ADAPTIVE_DOSE_ORACLE"))
        .collect();
    assert_eq!(
        w1,
        [
            &"W_ADAPTIVE_DOSE_ORACLE: 1 of 2 subject(s) were dosed from their simulated \
           individual values (dose_knowledge = oracle; start_dose reads CL). This is a \
           perfect-information arm: no prescriber has these values. Each such subject's \
           metrics have dose_oracle = true."
                .to_string()
        ]
    );

    for sim in [1, 2] {
        assert_eq!(
            first_dose(&r, "1", sim).to_bits(),
            dose_400_cl(70.0, 0.2).to_bits(),
            "supplied subject, replicate {sim}"
        );
    }
    let (d1, d2) = (first_dose(&r, "2", 1), first_dose(&r, "2", 2));
    let typical = dose_400_cl(50.0, 0.0);
    // The dose is 400 · the CL the subject is simulated under (no IOV, so one CL over the
    // whole run), backed out of its trough at 24 outside the engine (one bolus, no rule
    // fires). Worst relative gap measured on macOS arm64: 5.3e-16; bound 1e-12.
    for sim in [1, 2] {
        let d0 = first_dose(&r, "2", sim);
        let cl = backed_out_cl(&r, "2", sim, TVV * 50.0);
        let gap = ((d0 - 400.0 * cl) / (400.0 * cl)).abs();
        assert!(
            gap < 1e-12,
            "replicate {sim}: dose {d0} vs 400·CL {} ({gap:e})",
            400.0 * cl
        );
    }
    assert!(
        d1 != d2,
        "the oracle reads each replicate's draw: {d1} {d2}"
    );
    assert!(
        d1 != typical && d2 != typical,
        "{d1} {d2} vs η = 0 {typical}"
    );

    let r = run(&p, &population, &[("1", 0.2), ("2", -0.1)], 1).expect("all supplied");
    assert!(r.metrics.iter().all(|m| !m.dose_oracle));
    assert!(
        !r.warnings
            .iter()
            .any(|w| w.starts_with("W_ADAPTIVE_DOSE_ORACLE")),
        "{:?}",
        r.warnings
    );
}

/// The CL a subject ran under over `(0, 24]`, backed out of the trough at decision 24
/// outside the engine: one bolus `d₀` at 0 into a 1-cpt model with volume `v`, no rule
/// firing, so `C(24) = d₀/V · exp(−CL/V · 24)` and `CL = −V/24 · ln(C·V/d₀)`.
fn backed_out_cl(r: &AdaptiveSimulationResult, id: &str, sim: usize, v: f64) -> f64 {
    let d0 = first_dose(r, id, sim);
    let c = r
        .decisions
        .iter()
        .find(|d| d.subject == id && d.sim == sim && d.time == 24.0)
        .expect("decision at 24")
        .observed_signals[0]
        .value;
    assert!(c.is_finite() && c > 0.0, "{id} sim {sim}: trough {c}");
    -v / 24.0 * (c * v / d0).ln()
}

// ── IOV: κ = 0 for a supplied subject, κ₀ for an oracle one ───────────────────────

/// The κ the run draws for `id` in replicate `sim`, occasion `g`: `chol(Ω_IOV)·z` on the
/// subject's κ substream (the reconstruction `adaptive_auc_tv_tests` pins).
fn drawn_kappa(p: &ParsedModel, seed: u64, id: &str, sim: usize, g: usize) -> f64 {
    let chol = p
        .model
        .default_params
        .omega_iov
        .as_ref()
        .expect("Ω_IOV")
        .chol[(0, 0)];
    let base = crate::sim::adaptive::subject_kappa_base_seed(seed, id, sim);
    chol * crate::sim::adaptive::kappa_standard_normal(base, g, 0)
}

/// An IOV model (κ on CL, variance 0.5; η variance 0, so η = 0 exactly), one supplied
/// subject (η = 0) and one oracle subject, three replicates.
///
/// - The oracle subject's first dose is `400 · TVCL · exp(κ₀)`: it reads the first
///   decision's occasion, `decision_pk[0]`, the PK the driver resolves the dose's
///   attributes under — not κ = 0 and not the next occasion's κ₁.
/// - The supplied subject's first dose is `400 · TVCL` (κ = 0) in every replicate.
///
/// The κ values come from the run's own substream, so the reconstruction is checked first
/// against a quantity computed outside the engine: the supplied subject's CL backed out of
/// its trough at the second decision (one bolus, no rule fires, so
/// `CL = −V/24 · ln(C·V/d₀)`). With a record at each decision, the window `(0, 24]` is
/// governed by the record at 24 and runs under κ₁ (#943), so the back-out equals
/// `TVCL · exp(κ₁)` — which also shows κ is live for the supplied subject while its dose
/// ignores it. Worst relative gaps measured on macOS arm64: back-out 5.7e-16 (bound 1e-12),
/// oracle dose 0, so it is asserted to the bit.
///
/// Mutations: a supplied subject evaluated at `decision_pk[0]` (κ₀); an oracle subject
/// evaluated at κ = 0 (the baseline window), or at the next occasion's κ.
#[test]
fn under_iov_a_supplied_subject_reads_kappa_zero_and_an_oracle_one_the_first_occasion() {
    const SEED: u64 = 17;
    let b = block("400 * CL", "  dose_knowledge = oracle\n").replace("[0, 4000]", "[0, 1e6]");
    let p = model_with(true, "0", &b);
    let population = pop(&csv(&[("1", 70.0), ("2", 70.0)]));
    let r = run(&p, &population, &[("1", 0.0)], 3).expect("runs");
    let v = TVV * 70.0;
    let rel = |a: f64, b: f64| ((a - b) / b).abs();
    for sim in 1..=3 {
        // The reconstruction, against the back-out.
        let d0 = first_dose(&r, "1", sim);
        let c = r
            .decisions
            .iter()
            .find(|d| d.subject == "1" && d.sim == sim && d.time == 24.0)
            .expect("decision at 24")
            .observed_signals[0]
            .value;
        assert!(c.is_finite() && c > 0.0, "sim {sim}: trough {c}");
        let backed_out = -v / 24.0 * (c * v / d0).ln();
        let k1 = drawn_kappa(&p, SEED, "1", sim, 1);
        assert!(
            rel(backed_out, TVCL * k1.exp()) < 1e-12,
            "sim {sim}: backed-out CL {backed_out} vs TVCL·exp(κ₁) {}",
            TVCL * k1.exp()
        );
        assert!(rel(backed_out, TVCL) > 1e-2, "κ is live: CL {backed_out}");

        assert_eq!(d0, 400.0 * TVCL, "supplied sim {sim}: κ = 0");

        let oracle = first_dose(&r, "2", sim);
        let (k0, k1) = (
            drawn_kappa(&p, SEED, "2", sim, 0),
            drawn_kappa(&p, SEED, "2", sim, 1),
        );
        let want = 400.0 * (TVCL * (70.0 / 70.0) * (0.0 + k0).exp());
        assert_eq!(
            oracle.to_bits(),
            want.to_bits(),
            "oracle sim {sim}: {oracle} vs {want}"
        );
        for (label, other) in [("κ = 0", 0.0), ("κ₁", k1)] {
            let alt = 400.0 * TVCL * other.exp();
            assert!(
                rel(alt, want) > 1e-2,
                "sim {sim}: {label} must differ from κ₀"
            );
        }
    }
}

// ── The first decision: covariates and TIME ───────────────────────────────────────

/// A supplied subject whose WT changes before the first decision (t = 12) doses from CL
/// at the new WT; one whose WT changes after it doses from the old WT. Each side's dose is
/// `400 · CL(WT, η)` from the model's formula.
///
/// Mutations: the dose PK taken from the run's baseline `pk` (t = 0, the first WT);
/// re-evaluated at a later decision.
#[test]
fn an_individual_parameter_reads_the_covariate_in_force_at_the_first_decision() {
    let b = "  observe = CENT / V
  at = [12, 24, 36]
  start_dose = 400 * CL
  route = bolus(cmt=1)
  dose_bounds = [0, 4000]
  when signal > 1000000 : decrease 25%";
    let p = model(false, b);
    let csv = "ID,TIME,DV,AMT,EVID,CMT,MDV,WT
1,0,0,0,0,1,0,70
1,6,0,0,0,1,0,50
1,36,0,0,0,1,0,50
2,0,0,0,0,1,0,70
2,18,0,0,0,1,0,50
2,36,0,0,0,1,0,50
";
    let r = run(&p, &pop(csv), &[("1", 0.1), ("2", 0.1)], 1).expect("runs");
    let dose = |id: &str| {
        r.ledger
            .iter()
            .find(|e| e.subject == id && e.time == 12.0)
            .expect("a dose at 12")
            .amt
    };
    assert_eq!(dose("1").to_bits(), dose_400_cl(50.0, 0.1).to_bits());
    assert_eq!(dose("2").to_bits(), dose_400_cl(70.0, 0.1).to_bits());
}

/// CL reads `TIME`, and the first decision falls after a reset that restarts `TIME`
/// (written 0 at engine 25, so engine 31 is written 6). The dose reads CL at the written
/// `TIME` (6), the convention `decision_pk` and the monitors use (#1969, #2756), not the
/// engine clock (31). The premise — where the reader put the period — is asserted first.
///
/// Mutation: the dose PK evaluated at engine `at[0]` instead of `param_time_at(at[0])`.
#[test]
fn a_time_dependent_parameter_reads_the_written_time_at_the_first_decision() {
    let p = parse_full_model(
        "[parameters]
  theta TVCL(6.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 1.0
[individual_parameters]
  CL = TVCL * exp(ETA_CL) * (1 + 0.01 * TIME)
  V  = TVV
[structural_model]
  ode(states=[CENT])
[odes]
  d/dt(CENT) = -(CL / V) * CENT
[scaling]
  y = CENT / V
[error_model]
  DV ~ additive(ADD)
[adaptive_dosing]
  observe = CENT / V
  at = [31, 40]
  start_dose = 400 * CL
  route = bolus(cmt=1)
  dose_bounds = [0, 40000]
  when signal > 1000000 : decrease 25%",
    )
    .expect("model parses");
    let population = pop("ID,TIME,DV,EVID,AMT,CMT,MDV
1,0,0,1,100,1,1
1,6,0,0,0,1,0
1,24,0,0,0,1,0
1,0,0,3,0,1,1
1,0,0,1,100,1,1
1,6,0,0,0,1,0
");
    let s = &population.subjects[0];
    assert_eq!(s.reset_times, [25.0], "engine reset time");
    assert_eq!(
        s.param_time_at(31.0),
        6.0,
        "written TIME at the first decision"
    );
    let r = run(&p, &population, &[("1", 0.0)], 1).expect("runs");
    let d = r
        .ledger
        .iter()
        .find(|e| e.time == 31.0)
        .expect("a dose at 31")
        .amt;
    let at = |t: f64| 400.0 * (6.0 * 0.0f64.exp() * (1.0 + 0.01 * t));
    assert_eq!(d.to_bits(), at(6.0).to_bits(), "{d} vs written {}", at(6.0));
    assert!(d != at(31.0), "the engine clock would give {}", at(31.0));
}

// ── #2752: a model covariate missing from the in-force record ─────────────────────

/// An individual-parameter read gets its covariates through `pk_param_fn`, so a model
/// covariate missing from the snapshot of the record in force at the first decision would
/// read as 0 and dose from CL = 0. The run-start check (`check_covariates`, the #2752
/// predicate over every record map) refuses it before the dose is computed. Both sides:
/// the full population runs and doses `400 · CL(WT = 50)`; the gap is refused naming WT.
///
/// Mutation: an individual-parameter read routed around `check_simulation_data`.
#[test]
fn a_model_covariate_missing_from_the_in_force_record_is_refused_before_a_cl_read() {
    let b = "  observe = CENT / V
  at = [12, 24]
  start_dose = 400 * CL
  route = bolus(cmt=1)
  dose_bounds = [0, 4000]
  when signal > 1000000 : decrease 25%";
    let p = model(false, b);
    let full = pop("ID,TIME,DV,AMT,EVID,CMT,MDV,WT
1,0,0,0,0,1,0,70
1,6,0,0,0,1,0,50
1,24,0,0,0,1,0,50
");
    assert_eq!(
        full.subjects[0].obs_covariates.len(),
        3,
        "per-record snapshots"
    );
    let mut gap = full.clone();
    assert_eq!(gap.subjects[0].obs_covariates[1].remove("WT"), Some(50.0));
    let r = run(&p, &full, &[("1", 0.0)], 1).expect("every map carries WT");
    assert_eq!(
        first_dose_at(&r, 12.0).to_bits(),
        dose_400_cl(50.0, 0.0).to_bits()
    );
    let e = run(&p, &gap, &[("1", 0.0)], 1).expect_err("the in-force snapshot lacks WT");
    assert_eq!(e.code(), Some("E_MISSING_COVARIATE"), "{e}");
    assert!(e.message().contains("WT"), "{e}");
}

fn first_dose_at(r: &AdaptiveSimulationResult, t: f64) -> f64 {
    r.ledger.iter().find(|e| e.time == t).expect("a dose").amt
}

// ── E_ADAPTIVE_DOSE_VALUE for a subject-value read ────────────────────────────────

/// A start dose above the bounds, read from CL. A supplied subject is refused before any
/// subject runs, naming the subject and no replicate. An oracle subject is refused in its
/// first replicate, naming it. Both messages are asserted whole and name the CL value
/// read, which times 400 is the reported dose.
///
/// Mutations: the replicate dropped from the oracle message, or added to the supplied
/// one; the subject-value reads dropped from the "reads" sentence; a read reported from
/// another slot than the one the expression used.
#[test]
fn a_dose_value_refusal_names_the_replicate_and_the_individual_values_read() {
    let b = "  observe = CENT / V
  at = [0, 24]
  start_dose = 400 * CL
  route = bolus(cmt=1)
  dose_bounds = [0, 100]
  dose_knowledge = oracle
  when signal > 1000000 : decrease 25%";
    let p = model(false, b);

    let population = pop(&csv(&[("1", 70.0)]));
    let e = run(&p, &population, &[("1", 0.0)], 1).expect_err("supplied: 2400 > 100");
    assert_eq!(e.code(), Some("E_ADAPTIVE_DOSE_VALUE"), "{e}");
    assert_eq!(
        e.message(),
        "[adaptive_dosing] subject 1: start_dose = 400 * CL evaluates to 2400, outside \
         dose_bounds = [0, 100], which evaluate to [0, 100]. The dose expressions read \
         CL = 6, at the first decision (t = 0)."
    );

    // Subject 1 is supplied at WT 1 (400 · 6/70 ≈ 34 mg, within the bounds); subject 2
    // is drawn and reads its draw.
    let population = pop(&csv(&[("1", 1.0), ("2", 70.0)]));
    let e = run(&p, &population, &[("1", 0.0)], 1).expect_err("oracle: above 100");
    assert_eq!(e.code(), Some("E_ADAPTIVE_DOSE_VALUE"), "{e}");
    let msg = e.message();
    let head = "[adaptive_dosing] subject 2, replicate 1: start_dose = 400 * CL evaluates to ";
    assert!(msg.starts_with(head), "{msg}");
    let rest = &msg[head.len()..];
    let (dose, rest) = rest.split_once(", outside").expect("dose value");
    let cl = rest
        .strip_prefix(
            " dose_bounds = [0, 100], which evaluate to [0, 100]. The dose expressions read CL = ",
        )
        .and_then(|r| r.strip_suffix(", at the first decision (t = 0)."))
        .unwrap_or_else(|| panic!("{msg}"));
    let (dose, cl): (f64, f64) = (dose.parse().expect("dose"), cl.parse().expect("CL"));
    assert_eq!((400.0 * cl).to_bits(), dose.to_bits(), "{msg}");
    assert!(cl != TVCL, "the drawn CL, not the typical one: {cl}");
}

// ── C4 on a spec built in code ─────────────────────────────────────────────────────

/// `compile_dose_exprs` refuses an oracle spec whose dose expressions read no subject
/// value, as the parser does, so a spec built in code cannot be flagged as an oracle arm
/// that read no truth.
///
/// Mutation: the C4 check only at parse time.
#[test]
fn a_hand_built_oracle_spec_reading_no_subject_value_is_refused() {
    let p = model(false, &block("15 * WT", ""));
    let mut spec: AdaptiveDosingSpec = p.adaptive_dosing.clone().expect("block");
    spec.dose_knowledge = crate::sim::adaptive::DoseKnowledge::Oracle;
    let population = pop(&csv(&[("1", 70.0)]));
    let opts = AdaptiveSimulateOptions {
        seed: Some(7),
        ..Default::default()
    };
    let e = simulate_adaptive_from_spec(
        &p.model,
        &population,
        &p.model.default_params,
        1,
        &spec,
        &opts,
    )
    .expect_err("refused");
    assert_eq!(e.code(), Some("E_ADAPTIVE_SPEC"), "{e}");
    assert_eq!(
        e.message(),
        "[adaptive_dosing] dose_knowledge = oracle has no effect: no dose expression reads \
         an individual parameter, a random effect or MIXNUM."
    );
}

// ── An individual parameter no draw reaches is known for every subject (review r1 S1) ──

/// [`model`] with a typical value spelled as its own individual parameter, and one that
/// reaches η only through an intermediate.
fn model_tv(block: &str) -> ParsedModel {
    parse_full_model(&model_tv_src(block)).expect("model parses")
}

fn model_tv_src(block: &str) -> String {
    format!(
        "[parameters]
  theta TVCL({TVCL}, 0.1, 50.0)
  theta TVV({TVV}, 0.05, 5.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 1.0
[individual_parameters]
  TVCLW = TVCL * (WT / 70)
  CL = TVCLW * exp(ETA_CL)
  CL2 = 2 * CL
  V  = TVV * WT
[structural_model]
  ode(states=[CENT])
[odes]
  d/dt(CENT) = -(CL / V) * CENT
[scaling]
  y = CENT / V
[error_model]
  DV ~ additive(ADD)
[adaptive_dosing]
{block}"
    )
}

/// `TVCLW = TVCL * (WT / 70)` depends on θ and a covariate only, so every subject knows
/// it: with both subjects drawn under `known`, `400 * TVCLW` runs, doses
/// `400 · TVCL · WT/70` to the bit, flags no subject and does not warn. On the same model
/// and population, `400 * CL` (η directly) and `400 * CL2` (η only through `CL`) are
/// refused, and under `oracle` a block reading only `TVCLW` is C4 (nothing for the oracle
/// to apply to).
///
/// Mutations: every individual parameter classed as a subject value by name (the
/// `TVCLW` run is refused, and its oracle block passes C4 and flags every subject); the
/// classification not transitive (`CL2` let through, dosed from the draw); a draw-free
/// read evaluated with no PK (`TVCLW` reads 0, a 0 mg hold).
#[test]
fn a_typical_value_parameter_is_known_for_every_subject() {
    let population = pop(&csv(&[("1", 70.0), ("2", 50.0)]));
    let p = model_tv(&block("400 * TVCLW", ""));
    let r = run(&p, &population, &[], 1).expect("TVCLW is known for drawn subjects");
    for (id, wt) in [("1", 70.0), ("2", 50.0)] {
        let want = 400.0 * (TVCL * (wt / 70.0));
        assert_eq!(
            first_dose(&r, id, 1).to_bits(),
            want.to_bits(),
            "subject {id}"
        );
    }
    assert!(r.metrics.iter().all(|m| !m.dose_oracle && !m.eta_supplied));
    assert!(!r
        .warnings
        .iter()
        .any(|w| w.starts_with("W_ADAPTIVE_DOSE_ORACLE")));

    for (expr, name) in [("400 * CL", "CL"), ("400 * CL2", "CL2")] {
        let p = model_tv(&block(expr, ""));
        let e = run(&p, &population, &[], 1).expect_err("drawn");
        assert_eq!(e.code(), Some("E_ADAPTIVE_DOSE_KNOWLEDGE"), "{expr}: {e}");
        assert!(
            e.message().starts_with(&format!(
                "[adaptive_dosing] start_dose reads {name}, an individual parameter, but"
            )),
            "{expr}: {e}"
        );
    }

    let err = match parse_full_model(&model_tv_src(&block(
        "400 * TVCLW",
        "  dose_knowledge = oracle\n",
    ))) {
        Ok(_) => panic!("an oracle reading only TVCLW is refused"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.ends_with(
            "[adaptive_dosing] dose_knowledge = oracle has no effect: no dose expression \
             reads an individual parameter, a random effect or MIXNUM."
        ),
        "{err}"
    );
}

/// Two subject values read under different keys: the refusal names both, with the
/// plural wording, so the block is not refused one read at a time.
///
/// Mutation: the refusal built from `reads[0]` (or `reads.last()`) alone.
#[test]
fn a_knowledge_refusal_names_every_read() {
    let b = block("400 * CL", "").replace(
        "dose_bounds = [0, 4000]",
        "dose_bounds = [0, 4000 * exp(ETA_CL)]",
    );
    let p = model(false, &b);
    let e = run(&p, &pop(&csv(&[("1", 70.0), ("2", 50.0)])), &[], 1).expect_err("drawn");
    assert_eq!(e.code(), Some("E_ADAPTIVE_DOSE_KNOWLEDGE"), "{e}");
    assert_eq!(
        e.message(),
        "[adaptive_dosing] start_dose reads CL, an individual parameter, and dose_bounds \
         reads ETA_CL, a random effect, but the random effects of 2 of 2 subject(s) are \
         drawn by the simulation ('1', '2'), so these are the simulated truth, which no \
         prescriber knows at dosing time. To dose from them: carry the known values as data \
         columns (e.g. CL_KNOWN) and read those; supply each subject's random effects (e.g. \
         a previous fit's estimates, as `etas` in R, or through `AdaptiveInputs::with_etas` \
         and `simulate_adaptive_from_spec_with_inputs` in Rust); or set dose_knowledge = \
         oracle for a perfect-information comparison arm, which is flagged in the results."
    );
}

// ── MIXNUM under the oracle, at run time (review r1 S2) ──────────────────────────────

/// A two-class `[mixture]` model: class 2 clears 4× faster. `CLM` reaches the draw only
/// through `MIXNUM` (no η).
fn mixture_model(block: &str) -> ParsedModel {
    parse_full_model(&format!(
        "[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  theta MIXL(0.0, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04
[mixture]
  nsub = 2
  logit(1) = MIXL
[individual_parameters]
  CL = if (MIXNUM == 1) TVCL * exp(ETA_CL) else 4 * TVCL * exp(ETA_CL)
  CLM = if (MIXNUM == 1) TVCL else 4 * TVCL
  V  = TVV
[structural_model]
  ode(states=[CENT])
[odes]
  d/dt(CENT) = -(CL / V) * CENT
[scaling]
  y = CENT / V
[error_model]
  DV ~ proportional(PROP)
[adaptive_dosing]
{block}"
    ))
    .expect("model parses")
}

/// `start_dose = 100 * MIXNUM` under the oracle: each subject is dosed from the class it is
/// simulated under. The class is read back outside the engine from the CL backed out of the
/// trough (class 1 CL ≈ 5, class 2 ≈ 20 with ω² = 0.09; the threshold 10 sits between
/// them), and both classes occur. Under `known`, `100 * CLM` — reaching the draw only
/// through `MIXNUM` — is refused. A dose value refusal that reads `MIXNUM` names its value.
///
/// Mutations: the controller built outside the class guard (every subject dosed as class
/// 1); `MIXNUM` not counted as a draw (`CLM` let through); the `MIXNUM` arm of the reads
/// sentence reporting another value.
#[test]
fn an_oracle_reads_the_mixture_class_the_subject_is_simulated_under() {
    let ids: Vec<(String, f64)> = (1..=6).map(|i| (i.to_string(), 70.0)).collect();
    let ids: Vec<(&str, f64)> = ids.iter().map(|(i, w)| (i.as_str(), *w)).collect();
    let population = pop(&csv(&ids));
    let b = block("100 * MIXNUM", "  dose_knowledge = oracle\n");
    let p = mixture_model(&b);
    let r = run_drawn(&p, &population, 2).expect("oracle runs");
    let mut seen = [false, false];
    for (id, _) in &ids {
        for sim in [1, 2] {
            let d0 = first_dose(&r, id, sim);
            let cl = backed_out_cl(&r, id, sim, 50.0);
            let class = if cl > 10.0 { 2 } else { 1 };
            assert_eq!(d0, 100.0 * class as f64, "subject {id} sim {sim}: CL {cl}");
            seen[class - 1] = true;
        }
    }
    assert_eq!(seen, [true, true], "both classes occur");
    assert!(r.metrics.iter().all(|m| m.dose_oracle));

    let p = mixture_model(&block("100 * CLM", ""));
    let e = run_drawn(&p, &population, 1).expect_err("CLM reaches the draw through MIXNUM");
    assert_eq!(e.code(), Some("E_ADAPTIVE_DOSE_KNOWLEDGE"), "{e}");
    assert!(
        e.message()
            .starts_with("[adaptive_dosing] start_dose reads CLM, an individual parameter, but"),
        "{e}"
    );

    let b = block("100 * MIXNUM", "  dose_knowledge = oracle\n").replace("[0, 4000]", "[0, 150]");
    let e = run_drawn(&mixture_model(&b), &population, 1).expect_err("class 2 doses 200");
    assert_eq!(e.code(), Some("E_ADAPTIVE_DOSE_VALUE"), "{e}");
    let msg = e.message();
    let tail = ", replicate 1: start_dose = 100 * MIXNUM evaluates to 200, outside \
                dose_bounds = [0, 150], which evaluate to [0, 150]. The dose expressions \
                read MIXNUM = 2, at the first decision (t = 0).";
    assert!(
        msg.starts_with("[adaptive_dosing] subject ") && msg.ends_with(tail),
        "{msg}"
    );
}
