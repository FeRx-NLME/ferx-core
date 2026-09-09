//! `read_population_for_simulation` — a `DV = .` design template simulates
//! instead of reading as zero rows (#957).
//!
//! The fitting reader treats a missing DV on a scored row as a forgotten
//! `MDV=1` and skips it (#258), which is right when the DV is an input. On the
//! simulation path the DV is the *output*, so the same row is a design point.
//! These tests pin both readings of one template, and that the simulator
//! actually emits a row per retained sampling time.

use super::*;
use std::io::Write;

/// Dose plus three sampling times, `DV = .` everywhere — the natural way to
/// write a design, and what NONMEM's `$SIMULATION` accepts.
const TEMPLATE_CSV: &str = "ID,TIME,DV,EVID,AMT,CMT,MDV\n\
                            1,0,.,1,100,1,1\n\
                            1,0.25,.,0,.,1,0\n\
                            1,1,.,0,.,1,0\n\
                            1,4,.,0,.,1,0\n";

const SIM_MODEL: &str = "
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";

fn write_template() -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(TEMPLATE_CSV.as_bytes()).unwrap();
    f
}

#[test]
fn simulation_reader_keeps_a_dv_dot_template_the_fit_reader_drops() {
    let model = crate::parser::model_parser::parse_model_string(SIM_MODEL).expect("model parses");
    let f = write_template();
    let path = f.path().to_str().unwrap();

    let (sim_pop, _) =
        read_population_for_simulation(&model, &None, path, None, None, None, &[]).unwrap();
    assert_eq!(
        sim_pop.subjects[0].obs_times,
        vec![0.25, 1.0, 4.0],
        "the design's sampling times must survive the read"
    );
    assert_eq!(sim_pop.subjects[0].doses.len(), 1);

    // The fitting reader is unchanged: nothing to score, and it says so.
    let (fit_pop, _) = read_population_for(&model, &None, path, None, None, None, &[]).unwrap();
    assert!(fit_pop.subjects[0].obs_times.is_empty());
    assert!(fit_pop
        .warnings
        .iter()
        .any(|w| w.starts_with("W_MISSING_DV")));
}

#[test]
fn simulating_a_dv_dot_template_emits_one_row_per_sampling_time() {
    // The failure this closes: `ferx_simulate()` on a `DV = .` template returned
    // zero rows, and only a placeholder number in the column about to be
    // overwritten made it work.
    let model = crate::parser::model_parser::parse_model_string(SIM_MODEL).expect("model parses");
    let f = write_template();
    let path = f.path().to_str().unwrap();
    let (pop, _) =
        read_population_for_simulation(&model, &None, path, None, None, None, &[]).unwrap();

    let sims = simulate_with_seed(&model, &pop, &model.default_params, 1, 42);
    assert_eq!(sims.len(), 3, "one simulated row per sampling time");
    for (r, t) in sims.iter().zip([0.25, 1.0, 4.0]) {
        assert_eq!(r.time, t);
        assert!(r.ipred.is_finite() && r.ipred > 0.0, "ipred: {}", r.ipred);
        match r.outcome {
            SimOutcome::Continuous { value } => assert!(
                value.is_finite(),
                "the NaN DV placeholder must not leak into the simulated value"
            ),
            ref other => panic!("expected a continuous outcome, got {other:?}"),
        }
    }
}

// ── The design population is a simulation input, not a fit input ──────────
// `read_population_for_simulation`'s "do not pass this to `fit()`" contract used
// to be documentation-only, and the NaN placeholder was not the loud failure it
// was assumed to be (#957 review). These pin the three places that now catch it.

#[test]
fn the_simulation_reader_warns_that_it_kept_the_missing_dv_rows() {
    // A VPC simulated from an *observed* dataset keeps rows the fit skipped, so
    // the simulated dataset has more rows than the fitted one at times that were
    // never scored. Silent under either warning before; W_DESIGN_DV is the
    // simulation-side counterpart of W_MISSING_DV.
    let model = crate::parser::model_parser::parse_model_string(SIM_MODEL).expect("model parses");
    let f = write_template();
    let path = f.path().to_str().unwrap();

    let (sim_pop, _) =
        read_population_for_simulation(&model, &None, path, None, None, None, &[]).unwrap();
    let design: Vec<_> = sim_pop
        .warnings
        .iter()
        .filter(|w| w.starts_with("W_DESIGN_DV"))
        .collect();
    assert_eq!(
        design.len(),
        1,
        "exactly one summary: {:?}",
        sim_pop.warnings
    );
    assert!(
        design[0].contains("3 observation row(s)"),
        "the count must match the rows kept: {}",
        design[0]
    );
    assert!(
        !sim_pop
            .warnings
            .iter()
            .any(|w| w.starts_with("W_MISSING_DV")),
        "nothing was skipped on this path: {:?}",
        sim_pop.warnings
    );
}

#[test]
fn fitting_a_design_population_is_rejected_up_front() {
    // The misuse the contract only documented. Without E_NONFINITE_DV the NaN
    // placeholders reach the likelihood as a silent NaN objective.
    let model = crate::parser::model_parser::parse_model_string(SIM_MODEL).expect("model parses");
    let f = write_template();
    let path = f.path().to_str().unwrap();
    let (sim_pop, _) =
        read_population_for_simulation(&model, &None, path, None, None, None, &[]).unwrap();

    let diags = check_model_data(&model, &sim_pop);
    let d = diags
        .iter()
        .find(|d| d.code == "E_NONFINITE_DV")
        .unwrap_or_else(|| panic!("expected E_NONFINITE_DV, got {diags:?}"));
    assert!(d.message.contains("3 non-finite observation(s)"), "{d:?}");
    assert!(
        d.suggestion
            .as_deref()
            .unwrap_or_default()
            .contains("read_population_for"),
        "the suggestion must point at the fitting reader: {d:?}"
    );

    // The same dataset read for fitting is clean — the check keys on the DVs,
    // not on the dataset.
    let (fit_pop, _) = read_population_for(&model, &None, path, None, None, None, &[]).unwrap();
    assert!(
        !check_model_data(&model, &fit_pop)
            .iter()
            .any(|d| d.code == "E_NONFINITE_DV"),
        "no observations at all is not a non-finite observation"
    );
}

#[test]
fn ltbs_log_transform_does_not_launder_a_nan_placeholder() {
    // `f64::max` returns the non-NaN operand, so `NaN.max(LTBS_FLOOR).ln()` used
    // to turn every design point into a finite, extreme "observation" — the fit
    // then ran to completion on fabricated data with no warning.
    let model = crate::parser::model_parser::parse_model_string(SIM_MODEL).expect("model parses");
    let f = write_template();
    let path = f.path().to_str().unwrap();
    let (mut pop, _) =
        read_population_for_simulation(&model, &None, path, None, None, None, &[]).unwrap();

    let n_nonpos = log_transform_observations(&mut pop);
    assert_eq!(n_nonpos, 0, "a NaN is not a non-positive DV");
    assert!(
        pop.subjects[0].observations.iter().all(|v| v.is_nan()),
        "the placeholder must stay non-finite: {:?}",
        pop.subjects[0].observations
    );

    // Real values still transform, and the non-positive count is unchanged.
    let mut real = pop.clone();
    real.subjects[0].observations = vec![1.0_f64.exp(), -1.0, f64::NAN];
    assert_eq!(log_transform_observations(&mut real), 1);
    let obs = &real.subjects[0].observations;
    assert!((obs[0] - 1.0).abs() < 1e-12, "{obs:?}");
    assert!(
        obs[1].is_finite(),
        "a non-positive DV is still floored: {obs:?}"
    );
    assert!(obs[2].is_nan(), "{obs:?}");
}

#[test]
fn propensity_matching_rejects_a_design_template() {
    // Before #957 the reader dropped every row, so the "every subject needs
    // observations" precondition caught this. Now the rows exist with NaN DVs,
    // and the posthoc EBE would be optimized against a NaN objective.
    let model = crate::parser::model_parser::parse_model_string(SIM_MODEL).expect("model parses");
    let f = write_template();
    let path = f.path().to_str().unwrap();
    let (pop, _) =
        read_population_for_simulation(&model, &None, path, None, None, None, &[]).unwrap();

    let opts = SimulateOptions {
        seed: Some(1),
        match_method: Some(crate::propensity_match::MatchMethod::Optimal),
        ..Default::default()
    };
    let err = simulate_with_options(&model, &pop, &model.default_params, 1, &opts).unwrap_err();
    assert!(
        err.contains("non-finite DV values") && err.contains("design template"),
        "the message must name the real cause, not an EBE convergence failure: {err}"
    );

    // Matching off, the same population simulates as before.
    let ok = SimulateOptions {
        seed: Some(1),
        ..Default::default()
    };
    assert_eq!(
        simulate_with_options(&model, &pop, &model.default_params, 1, &ok)
            .unwrap()
            .len(),
        3
    );
}
