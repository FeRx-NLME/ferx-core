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
