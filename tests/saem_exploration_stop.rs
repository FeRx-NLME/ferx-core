//! Tier-2: SAEM's exploration phase ends by a convergence test unless
//! `n_exploration` fixes it, and the fit reports the length it used.
//!
//! The rule itself (trend test, window, calibration) is unit-tested in
//! `estimation/saem_exploration_tests.rs`; these tests pin the wiring through
//! `fit()`: precedence of the explicit length, the minimum/maximum, the
//! `FitResult::saem_n_exploration_used` report, the maximum-reached warning, and
//! the smoothing exponent. Every fit returns after at most ~700 iterations of a
//! 40-subject model.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, OmegaMatrix, Population};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions, FitResult};

mod common;

const MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  1.0, 500.0)
  omega ETA_CL ~ 0.10
  omega ETA_V  ~ 0.10
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn population(model: &ferx_core::types::CompiledModel) -> Population {
    let times = vec![0.5_f64, 1.0, 2.0, 4.0, 8.0, 12.0];
    let subjects = (1..=40)
        .map(|i| {
            common::subject(
                &format!("{i}"),
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                times.clone(),
                vec![0.0; times.len()],
                vec![1; times.len()],
            )
        })
        .collect();
    let template = Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let mut truth = model.default_params.clone();
    truth.theta = vec![5.0, 50.0];
    truth.omega = OmegaMatrix::from_diagonal(&[0.09, 0.09], vec!["ETA_CL".into(), "ETA_V".into()]);
    truth.sigma.values = vec![0.10];
    let sim = simulate_with_seed(model, &template, &truth, 1, 20240527).unwrap();
    let mut pop = template.clone();
    for subj in pop.subjects.iter_mut() {
        subj.observations = sim
            .iter()
            .filter(|s| s.id == subj.id)
            .map(|s| s.outcome.continuous_value().max(1e-6))
            .collect();
    }
    pop
}

fn run(edit: impl FnOnce(&mut FitOptions)) -> FitResult {
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = population(&model);
    let mut o = FitOptions::default();
    o.method = EstimationMethod::Saem;
    o.saem_n_convergence = 10;
    o.saem_seed = Some(11);
    o.run_covariance_step = false;
    o.verbose = false;
    edit(&mut o);
    fit(&model, &pop, &model.default_params, &o).expect("SAEM fit must succeed")
}

fn max_warning(r: &FitResult) -> bool {
    r.warnings.iter().any(|w| w.contains("exploration phase reached its maximum"))
}

/// An explicit `n_exploration` is a fixed length: it is used as given, the
/// min/max are ignored (here deliberately set to contradict it), and no
/// maximum-reached warning is raised.
#[test]
fn explicit_n_exploration_is_a_fixed_length() {
    let r = run(|o| {
        o.saem_n_exploration = Some(12);
        o.saem_exploration_min = 60;
        o.saem_exploration_max = 90;
    });
    assert_eq!(r.saem_n_exploration_used, Some(12));
    assert!(!max_warning(&r), "{:?}", r.warnings);
}

/// The automatic stop never ends before the minimum and never exceeds the
/// maximum. With `max = min = 15` no coordinate can be tested (the shortest test
/// window is 100 iterations), so the run is cut off by the maximum and says so.
#[test]
fn maximum_is_respected_and_reported() {
    let r = run(|o| {
        o.saem_exploration_min = 15;
        o.saem_exploration_max = 15;
    });
    assert_eq!(r.saem_n_exploration_used, Some(15));
    assert!(max_warning(&r), "{:?}", r.warnings);
    // The max-reached warning names the coordinates not shown stationary (here all of
    // them, untestable), and a separate warning says the settings make the test impossible.
    assert!(
        r.warnings
            .iter()
            .any(|w| w.contains("reached its maximum") && w.contains("complete-data -logL")),
        "{:?}",
        r.warnings
    );
    assert!(r.warnings.iter().any(|w| w.contains("can never be tested")), "{:?}", r.warnings);
}

/// Well-identified model: the rule needs ten points per coordinate at a spacing of
/// at least 10, so the run ends between the shortest window (100) and the maximum,
/// on a check iteration, and without the maximum-reached warning.
#[test]
fn automatic_stop_lies_between_the_bounds() {
    let r = run(|o| {
        o.saem_exploration_min = 20;
        o.saem_exploration_max = 700;
    });
    let used = r.saem_n_exploration_used.expect("SAEM reports its exploration length");
    // Requiring a stop strictly before the maximum is what makes the test fail if the
    // automatic stop is never taken (which would also yield a length inside the bounds).
    eprintln!("automatic stop used {used}");
    assert!((100..700).contains(&used), "used {used}");
    assert_eq!(used % 10, 0, "stops only on a check iteration, used {used}");
    assert!(!max_warning(&r), "{:?}", r.warnings);
}

/// The report is SAEM-only and survives to the result of the default
/// configuration (automatic, 150..2000).
#[test]
fn default_is_automatic_and_non_saem_reports_nothing() {
    assert!(FitOptions::default().saem_n_exploration.is_none());
    let model = parse_model_string(MODEL).unwrap();
    let pop = population(&model);
    let mut o = FitOptions::default();
    o.method = EstimationMethod::FoceI;
    o.run_covariance_step = false;
    o.outer_maxiter = 3;
    let r = fit(&model, &pop, &model.default_params, &o).unwrap();
    assert_eq!(r.saem_n_exploration_used, None);
}

/// `smoothing_exponent = 1` is the classical schedule bit for bit; `0.7`
/// changes the convergence-phase averaging and therefore the estimates.
#[test]
fn smoothing_exponent_one_is_the_default_and_seven_tenths_differs() {
    let fixed = |o: &mut FitOptions| {
        o.saem_n_exploration = Some(10);
        o.saem_n_convergence = 15;
    };
    let base = run(fixed);
    let one = run(|o| {
        fixed(o);
        o.saem_smoothing_exponent = 1.0;
    });
    let sev = run(|o| {
        fixed(o);
        o.saem_smoothing_exponent = 0.7;
    });
    assert_eq!(base.theta, one.theta);
    assert_eq!(base.omega.diagonal(), one.omega.diagonal());
    assert_ne!(base.theta, sev.theta);
}

/// A `FitOptions` built in code bypasses the model-file parser's range check;
/// `run_saem` rejects the exponent itself rather than averaging with a
/// non-decaying (0), growing (negative) or NaN step.
#[test]
fn smoothing_exponent_out_of_range_is_rejected_from_the_api() {
    let model = parse_model_string(MODEL).expect("model must parse");
    let pop = population(&model);
    for bad in [0.0, 0.5, -1.0, 1.5, f64::NAN] {
        let mut o = FitOptions::default();
        o.method = EstimationMethod::Saem;
        o.saem_n_exploration = Some(5);
        o.saem_n_convergence = 5;
        o.saem_seed = Some(11);
        o.run_covariance_step = false;
        o.verbose = false;
        o.saem_smoothing_exponent = bad;
        let err = fit(&model, &pop, &model.default_params, &o)
            .err()
            .unwrap_or_else(|| panic!("exponent {bad} must be rejected"));
        let msg = err.to_string();
        assert!(msg.contains("smoothing_exponent"), "{bad}: {msg}");
    }
}

/// Two chains per subject are the default (NONMEM `ISAMPLE=2`). The chains are
/// an internal replication of the subjects for the SAEM iterations only: the
/// result has one row per *data* subject with the data's ids, and a different
/// chain count gives a different fit on the same seed (so the option is live).
#[test]
fn two_chains_are_the_default_and_the_result_keeps_the_data_subjects() {
    assert_eq!(FitOptions::default().saem_n_chains, 2);
    let fixed = |o: &mut FitOptions| o.saem_n_exploration = Some(30);
    let two = run(fixed);
    let one = run(|o| {
        fixed(o);
        o.saem_n_chains = 1;
    });
    let model = parse_model_string(MODEL).unwrap();
    let pop = population(&model);
    for r in [&two, &one] {
        assert_eq!(r.subjects.len(), pop.subjects.len());
        for (s, d) in r.subjects.iter().zip(&pop.subjects) {
            assert_eq!(s.id, d.id, "no chain copy leaks into the result");
        }
    }
    assert_ne!(two.theta, one.theta, "the chain count changes the fit");
}

/// #2986: `saem_kernel = nonmem` runs NONMEM's η kernels (independence proposals from the
/// population density and from the running conditional moments, then the block and
/// componentwise random walks). It must reach the E-step — a different fit from the default
/// random walk on the same seed — and recover the simulated CL and V as well.
#[test]
fn nonmem_kernels_fit_and_differ_from_the_random_walk() {
    let fixed = |o: &mut FitOptions| {
        o.saem_n_exploration = Some(60);
        o.saem_n_convergence = 60;
    };
    assert_eq!(FitOptions::default().saem_kernel, ferx_core::types::SaemKernel::Hmc);
    let rw = run(|o| {
        fixed(o);
        o.saem_kernel = ferx_core::types::SaemKernel::RandomWalk;
    });
    let nm = run(|o| {
        fixed(o);
        o.saem_kernel = ferx_core::types::SaemKernel::Nonmem;
    });
    assert_ne!(rw.theta, nm.theta, "the kernel option reaches the E-step");
    for (name, got, truth) in [("CL", nm.theta[0], 5.0), ("V", nm.theta[1], 50.0)] {
        assert!((got / truth - 1.0).abs() < 0.15, "{name}: {got} vs simulated {truth}");
    }
}

/// #2986: `saem_kernel = hmc | nuts` replace NONMEM's kernels 2 and 3 by a gradient move whose
/// η-gradient the AD engine gives. On this closed-form model it must give one for every subject
/// (no "gave no η-gradient" warning), the move must reach the E-step (a different fit from
/// NONMEM's kernels on the same seed), and the fit must recover the simulated CL and V.
#[test]
fn gradient_kernels_are_served_differ_and_recover_the_truth() {
    use ferx_core::types::SaemKernel;
    let fixed = |o: &mut FitOptions| {
        o.saem_n_exploration = Some(60);
        o.saem_n_convergence = 60;
    };
    let nm = run(|o| {
        fixed(o);
        o.saem_kernel = SaemKernel::Nonmem;
    });
    for kernel in [SaemKernel::Hmc, SaemKernel::Nuts] {
        let r = run(|o| {
            fixed(o);
            o.saem_kernel = kernel;
        });
        assert!(
            !r.warnings.iter().any(|w| w.contains("gave no subject's η-gradient")),
            "{kernel:?}: the AD engine serves this model: {:?}",
            r.warnings
        );
        assert_ne!(r.theta, nm.theta, "{kernel:?} reaches the E-step");
        let want = if kernel == SaemKernel::Hmc { "imh 1, hmc 1" } else { "imh 1, nuts 1" };
        assert_eq!(r.saem_kernels.as_deref(), Some(want), "the fit reports the schedule it ran");
        let g = r.saem_gradient_kernel.as_ref().expect("gradient diagnostics");
        assert!(g.transitions > 0 && g.declined == 0, "{g:?}");
        for (name, got, truth) in [("CL", r.theta[0], 5.0), ("V", r.theta[1], 50.0)] {
            assert!(
                (got / truth - 1.0).abs() < 0.15,
                "{kernel:?} {name}: {got} vs simulated {truth}"
            );
        }
    }
}

/// #2986: under IOV the gradient move is joint over `[η, κ₁..κ_K]`, on the stacked IOV gradient
/// the AD engine gives. `warfarin_iov_saem` (three η, one κ over occasions) must be served — no
/// "gave no η-gradient" warning — and fit to a finite objective different from NONMEM's kernels.
#[test]
fn the_gradient_kernel_serves_an_iov_model() {
    use ferx_core::types::SaemKernel;
    let prep = ferx_core::prepare_run(
        "examples/warfarin_iov_saem.ferx",
        Some("data/warfarin_iov_saem.csv"),
    )
    .expect("prepare");
    let fit_with = |kernel: SaemKernel| {
        let mut o = prep.parsed.fit_options.clone();
        o.saem_kernel = kernel;
        o.saem_n_exploration = Some(40);
        o.saem_n_convergence = 40;
        o.saem_ofv = ferx_core::types::SaemOfv::Focei;
        o.run_covariance_step = false;
        o.verbose = false;
        o.threads = Some(1);
        fit(&prep.parsed.model, &prep.population, &prep.init_params, &o).expect("fit")
    };
    let hmc = fit_with(SaemKernel::Hmc);
    let nm = fit_with(SaemKernel::Nonmem);
    assert!(
        !hmc.warnings.iter().any(|w| w.contains("gave no subject's η-gradient")),
        "the IOV gradient is served: {:?}",
        hmc.warnings
    );
    assert!(hmc.ofv.is_finite());
    assert_ne!(hmc.theta, nm.theta, "the joint move reaches the E-step");
}

/// #2986: a user picks the kernels. A combination no preset names — f-SAEM jumps, HMC and a
/// componentwise sweep — runs, is reported in its order, and differs from HMC alone; the
/// default reports HMC's preset and `nonmem` NONMEM's; every count at 0 is refused with a message naming the
/// counts.
#[test]
fn a_user_chosen_kernel_combination_runs_and_all_zero_is_refused() {
    use ferx_core::types::SaemKernel;
    let fixed = |o: &mut FitOptions| {
        o.saem_n_exploration = Some(40);
        o.saem_n_convergence = 40;
    };
    let nm = run(|o| {
        fixed(o);
        o.saem_kernel = SaemKernel::Nonmem;
    });
    assert_eq!(nm.saem_kernels.as_deref(), Some("m1 2, m1b 2, m2 2, m3 2"));
    assert_eq!(run(fixed).saem_kernels.as_deref(), Some("imh 1, hmc 1"), "the default preset");
    let hmc = run(|o| {
        fixed(o);
        o.saem_kernel = SaemKernel::Hmc;
    });
    let mix = run(|o| {
        fixed(o);
        o.saem_kernel = SaemKernel::Hmc;
        o.saem_isample_imh = Some(0);
        o.saem_isample_m3 = Some(1);
    });
    assert_eq!(mix.saem_kernels.as_deref(), Some("hmc 1, m3 1"));
    assert_ne!(mix.theta, hmc.theta, "the added kernels reach the E-step");
    let model = parse_model_string(MODEL).expect("parses");
    let pop = population(&model);
    let mut o = FitOptions {
        method: EstimationMethod::Saem,
        saem_isample_m1: Some(0),
        saem_isample_imh: Some(0),
        saem_isample_m1b: Some(0),
        saem_isample_hmc: Some(0),
        saem_isample_nuts: Some(0),
        saem_isample_m2: Some(0),
        saem_isample_m3: Some(0),
        ..FitOptions::default()
    };
    fixed(&mut o);
    let err = fit(&model, &pop, &model.default_params, &o).expect_err("no kernel moves η");
    assert!(format!("{err}").contains("every E-step kernel count is 0"), "{err}");
}
