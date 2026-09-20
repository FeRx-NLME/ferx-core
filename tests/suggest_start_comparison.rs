//! Comparison test: inits_from_nca with NcaInit::Nca (A), ::Sweep (B), ::Ebe (C)
//! across analytical and ODE model types.
//!
//! Run with:
//!   cargo test --no-default-features --features ci suggest_start_comparison -- --nocapture

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{inits_from_nca, read_nonmem_csv, NcaInit};
use std::path::Path;
use std::time::Instant;

struct Case {
    label: &'static str,
    model_path: &'static str,
    data_path: &'static str,
    /// True / expected values for each theta (None = no ground truth available)
    truth: Vec<(&'static str, f64)>,
}

fn run_case(case: &Case) {
    let model = parse_model_file(Path::new(case.model_path))
        .unwrap_or_else(|e| panic!("parse {}: {e}", case.model_path));
    let population = read_nonmem_csv(Path::new(case.data_path), None, None)
        .unwrap_or_else(|e| panic!("data {}: {e}", case.data_path));

    let t0 = Instant::now();
    let a = inits_from_nca(&model, &population, NcaInit::Nca).unwrap();
    let t_a = t0.elapsed();

    let t1 = Instant::now();
    let b = inits_from_nca(&model, &population, NcaInit::Sweep).unwrap();
    let t_b = t1.elapsed();

    let t2 = Instant::now();
    let c = inits_from_nca(&model, &population, NcaInit::Ebe).unwrap();
    let t_c = t2.elapsed();

    println!("\n══ {} ══", case.label);
    println!(
        "  Timing: A={:.0}ms  B={:.0}ms  C={:.0}ms",
        t_a.as_secs_f64() * 1000.0,
        t_b.as_secs_f64() * 1000.0,
        t_c.as_secs_f64() * 1000.0,
    );
    println!("  Theta              Default    OptA       OptB       OptC       Truth");
    println!("  {:-<72}", "");

    for (i, name) in a.params.theta_names.iter().enumerate() {
        let default = model.default_params.theta[i];
        let opt_a = a.params.theta[i];
        let opt_b = b.params.theta[i];
        let opt_c = c.params.theta[i];
        let truth_str = case
            .truth
            .iter()
            .find(|(n, _)| *n == name.as_str())
            .map(|(_, v)| format!("{v:>10.3}"))
            .unwrap_or_else(|| "          ".to_string());
        let changed_a = (opt_a - default).abs() > 1e-10;
        let changed_b = (opt_b - opt_a).abs() > 1e-10;
        let changed_c = (opt_c - opt_b).abs() > 1e-10;
        let tag_a = if changed_a { "*" } else { " " };
        let tag_b = if changed_b { "†" } else { " " };
        let tag_c = if changed_c { "‡" } else { " " };
        println!(
            "  {name:<18} {default:>10.3} {opt_a:>10.3}{tag_a} {opt_b:>10.3}{tag_b} {opt_c:>10.3}{tag_c}{truth_str}"
        );
    }

    println!("  (* = A changed default; † = B changed A; ‡ = C changed B)");

    if !a.warnings.is_empty() {
        println!("  Option A warnings:");
        for w in &a.warnings {
            println!("    [A] {w}");
        }
    }
    let b_only: Vec<_> = b
        .warnings
        .iter()
        .filter(|w| !a.warnings.contains(w))
        .collect();
    if !b_only.is_empty() {
        println!("  Option B additional warnings:");
        for w in b_only {
            println!("    [B] {w}");
        }
    }
    let c_only: Vec<_> = c
        .warnings
        .iter()
        .filter(|w| !a.warnings.contains(w) && !b.warnings.contains(w))
        .collect();
    if !c_only.is_empty() {
        println!("  Option C additional warnings:");
        for w in c_only {
            println!("    [C] {w}");
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn suggest_start_comparison_all_models() {
    let cases = vec![
        // ── Analytical models ────────────────────────────────────────────────
        Case {
            label: "1-cpt IV bolus (analytical)",
            model_path: "examples/one_cpt_iv.ferx",
            data_path: "data/one_cpt_iv.csv",
            truth: vec![("TVCL", 5.0), ("TVV", 50.0)],
        },
        Case {
            label: "1-cpt oral (analytical, warfarin)",
            model_path: "examples/warfarin.ferx",
            data_path: "data/warfarin.csv",
            truth: vec![("TVCL", 0.134), ("TVV", 8.0), ("TVKA", 1.0)],
        },
        Case {
            label: "2-cpt IV bolus (analytical)",
            model_path: "examples/two_cpt_iv.ferx",
            data_path: "data/two_cpt_iv.csv",
            truth: vec![("TVCL", 4.0), ("TVV1", 12.0), ("TVQ", 2.0), ("TVV2", 25.0)],
        },
        Case {
            label: "2-cpt oral (analytical, with covariates)",
            model_path: "examples/two_cpt_oral_cov.ferx",
            data_path: "data/two_cpt_oral_cov.csv",
            truth: vec![("TVCL", 4.0), ("TVV1", 40.0), ("TVKA", 1.0)],
        },
        Case {
            label: "3-cpt IV bolus (analytical)",
            model_path: "examples/three_cpt_iv.ferx",
            data_path: "data/three_cpt_iv.csv",
            truth: vec![
                ("TVCL", 5.0),
                ("TVV1", 10.0),
                ("TVQ2", 2.0),
                ("TVV2", 20.0),
                ("TVQ3", 1.5),
                ("TVV3", 30.0),
            ],
        },
        Case {
            label: "1-cpt oral with lag time (analytical)",
            model_path: "examples/oral_with_lagtime.ferx",
            data_path: "data/one_cpt_oral_lagtime.csv",
            truth: vec![
                ("TVCL", 5.0),
                ("TVV", 50.0),
                ("TVKA", 1.5),
                ("TVLAGTIME", 0.75),
            ],
        },
        Case {
            label: "1-cpt oral with bioavailability F (warfarin_logit_f)",
            model_path: "examples/warfarin_logit_f.ferx",
            data_path: "data/warfarin_logit_f.csv",
            // THETA_F default 0.80; NCA CL/V are apparent (CL/F, V/F) and are
            // corrected by F_default so TVCL/TVV start in the true-parameter space.
            // F itself stays at default for Option A; Option B sweeps it.
            truth: vec![
                ("TVCL", 0.134),
                ("TVV", 8.1),
                ("TVKA", 1.0),
                ("THETA_F", 0.80),
            ],
        },
        Case {
            label: "1-cpt IV infusion (analytical)",
            model_path: "examples/one_cpt_infusion.ferx",
            data_path: "data/one_cpt_infusion.csv",
            truth: vec![("TVCL", 5.0), ("TVV", 50.0)],
        },
        // ── ODE models ───────────────────────────────────────────────────────
        Case {
            label: "1-cpt IV bolus (ODE)",
            model_path: "examples/one_cpt_iv_ode.ferx",
            data_path: "data/one_cpt_iv.csv",
            truth: vec![("TVCL", 5.0), ("TVV", 50.0)],
        },
        Case {
            label: "1-cpt oral (ODE, warfarin)",
            model_path: "examples/warfarin_ode.ferx",
            data_path: "data/warfarin.csv",
            truth: vec![("TVCL", 0.134), ("TVV", 8.0), ("TVKA", 1.0)],
        },
        Case {
            label: "2-cpt IV bolus (ODE)",
            model_path: "examples/two_cpt_iv_ode.ferx",
            data_path: "data/two_cpt_iv.csv",
            truth: vec![("TVCL", 4.0), ("TVV1", 12.0), ("TVQ", 2.0), ("TVV2", 25.0)],
        },
        Case {
            label: "2-cpt oral (ODE)",
            model_path: "examples/two_cpt_oral_ode.ferx",
            data_path: "data/two_cpt_oral_cov.csv",
            truth: vec![("TVCL", 4.0), ("TVV1", 40.0), ("TVKA", 1.0)],
        },
        Case {
            label: "3-cpt IV bolus (ODE)",
            model_path: "examples/three_cpt_iv_ode.ferx",
            data_path: "data/three_cpt_iv.csv",
            truth: vec![("TVCL", 5.0), ("TVV1", 10.0)],
        },
        // ── Name-independence & non-standard model types ─────────────────────
        Case {
            label: "1-cpt IV bolus (renamed: CLEARANCE/DISTRIB)",
            model_path: "examples/one_cpt_iv_renamed.ferx",
            data_path: "data/one_cpt_iv.csv",
            // Thetas are named CLEARANCE and DISTRIB — find_theta_for_slot must
            // resolve them through mu_refs rather than canonical name search.
            truth: vec![("CLEARANCE", 5.0), ("DISTRIB", 50.0)],
        },
        Case {
            label: "MM IV bolus (Michaelis-Menten, ODE)",
            model_path: "examples/mm_iv.ferx",
            data_path: "data/mm_iv.csv",
            // VMAX and KM do not match any canonical PK slot name —
            // Option A returns model defaults; Option B sweeps all thetas via
            // 1D coordinate sweeps.  No CL/V ridge, so 1D sweeps are appropriate.
            truth: vec![("TVVMAX", 5.0), ("TVKM", 2.0), ("TVV", 10.0)],
        },
    ];

    for case in &cases {
        run_case(case);
    }

    println!("\nLegend: * = A wrote value; † = B refined over A; ‡ = C (EBE) refined over B");
}
