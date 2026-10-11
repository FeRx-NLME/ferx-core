//! An individual parameter assigned on *every* arm of an `if`/`else` and read
//! only by an ODE `[event_model] hazard =` (the #357 construction, for an
//! endpoint). The hazard is folded into the `[odes]` RHS as a cumulative-hazard
//! state, and that RHS resolves individual parameters only, so the name has to
//! earn a slot exactly as one `[odes]` reads does. Before the fix the model was
//! refused with "[odes]: RHS references undefined name(s): H0".
//!
//! The check is an equivalence against the top-level ternary spelling of the
//! same value, `H0 = if (WT > 70) A else B`, which was always an individual
//! parameter. Two subjects straddle the condition, so both arms are live, and a
//! constant hazard gives a closed form, `H(t) = h·t`, that pins the value
//! independently of either spelling.

#![cfg(feature = "survival")]

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{CompiledModel, Population};
use std::io::Write;

/// Hazard 0.01 above WT 70, 0.02 at or below.
const ALL_ARM: &str = "  if (WT > 70) {\n    H0 = 0.01\n  } else {\n    H0 = 0.02\n  }";
const TERNARY: &str = "  H0 = if (WT > 70) 0.01 else 0.02";

/// Subject 1: WT 60 (the `else` arm). Subject 2: WT 80 (the `if` arm). One
/// dose and one right-censored TTE record at t = 5.
const DATA: &str = "\
ID,TIME,DV,EVID,AMT,CMT,MDV,WT
1,0,.,1,100,1,1,60
1,5,0,0,.,2,0,60
2,0,.,1,100,1,1,80
2,5,0,0,.,2,0,80
";

fn model(ip: &str, header: &str) -> String {
    format!(
        "[parameters]\n  theta TVCL(3.0, 0.01, 100.0)\n  theta TVV(20.0, 1.0, 500.0)\n  \
         omega ETA_CL ~ 0.1\n  sigma PROP_ERR ~ 0.04\n\
         [individual_parameters]\n  V = TVV\n  CL = TVCL * exp(ETA_CL)\n{ip}\n\
         [structural_model]\n  ode(obs_cmt=central, states=[central])\n\
         [odes]\n  d/dt(central) = -(CL/V) * central\n\
         [scaling]\n  obs_scale = V\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n\
         {header}\n  cmt    = 2\n  hazard = H0\n"
    )
}

fn load(src: &str) -> (CompiledModel, Population) {
    let m = parse_model_string(src).unwrap_or_else(|e| panic!("model must parse: {e}"));
    let mut f = tempfile::NamedTempFile::new().expect("temp csv");
    f.write_all(DATA.as_bytes()).expect("write csv");
    let (pop, _) =
        read_population_for(&m, &None, f.path().to_str().unwrap(), None, None, None, &[])
            .unwrap_or_else(|e| panic!("data must load: {e}"));
    (m, pop)
}

fn cum_hazard_at_5(src: &str) -> Vec<f64> {
    let (m, pop) = load(src);
    ferx_core::predict_survival(&m, &pop, &m.default_params, &[5.0])
        .expect("predict_survival")
        .iter()
        .map(|r| r.cum_hazard)
        .collect()
}

#[test]
fn all_arm_hazard_parameter_predicts_like_its_ternary_twin() {
    for header in ["[event_model]", "[event_model DEATH]"] {
        let all_arm = cum_hazard_at_5(&model(ALL_ARM, header));
        let ternary = cum_hazard_at_5(&model(TERNARY, header));
        assert_eq!(
            all_arm, ternary,
            "{header}: the two spellings must agree bit for bit"
        );
        // Closed form, H(5) = h · 5: 0.02 · 5 for WT 60, 0.01 · 5 for WT 80.
        // Both arms are live: the two subjects differ by a factor of two.
        assert_eq!(all_arm.len(), 2, "{header}: one row per subject");
        assert!(
            (all_arm[0] - 0.10).abs() < 1e-9,
            "{header}: WT 60 H(5) = {}",
            all_arm[0]
        );
        assert!(
            (all_arm[1] - 0.05).abs() < 1e-9,
            "{header}: WT 80 H(5) = {}",
            all_arm[1]
        );
    }
}
