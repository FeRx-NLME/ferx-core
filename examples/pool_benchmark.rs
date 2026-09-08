//! Reproducible pool-lifecycle / real-fit benchmark. Run with the same profile on
//! both revisions: cargo run --profile ci-test --example pool_benchmark -- LABEL
//! Optional second argument: number of measured rounds (default 7).
use ferx_core::{fit, parser::model_parser::parse_model_string, types::*, PoolPlan};
use rayon::prelude::*;
use std::{collections::HashMap, time::Instant};

fn model(ode: bool) -> CompiledModel {
    let structural = if ode {
        "ode(states=[central])\n[odes]\n d/dt(central) = -(CL / V) * central\n[scaling]\n y = central / V"
    } else {
        "pk one_cpt_iv(cl=CL, v=V)"
    };
    parse_model_string(&format!(
        "[parameters]\n theta TVCL(1.0, 0.1, 50.0)\n theta TVV(10.0, 1.0, 500.0)\n omega ETA_CL ~ 0.04\n sigma PROP ~ 0.04\n[individual_parameters]\n CL = TVCL * exp(ETA_CL)\n V = TVV\n[structural_model]\n {structural}\n[error_model]\n DV ~ proportional(PROP)"
    )).unwrap()
}

fn population(n: usize) -> Population {
    Population {
        subjects: (0..n)
            .map(|i| {
                let obs_times = vec![0.5, 2.0, 8.0, 24.0];
                let observations = obs_times
                    .iter()
                    .enumerate()
                    .map(|(j, &t)| {
                        let cl = 1.1 * (0.2 * ((i * 7 % 13) as f64 / 6.0 - 1.0)).exp();
                        10.0 * (-cl * t / 10.0).exp()
                            * (1.0 + 0.08 * ((i + j * 3) % 7) as f64 - 0.24)
                    })
                    .collect();
                Subject {
                    id: (i + 1).to_string(),
                    doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                    obs_times,
                    observations,
                    obs_cmts: vec![1; 4],
                    cens: vec![0; 4],
                    obs_raw_times: vec![],
                    covariates: HashMap::new(),
                    dose_covariates: vec![],
                    obs_covariates: vec![],
                    pk_only_times: vec![],
                    pk_only_covariates: vec![],
                    reset_times: vec![],
                    reset_covariates: vec![],
                    occasions: vec![],
                    obs_l2: vec![],
                    dose_occasions: vec![],
                    reset_occasions: vec![],
                    fremtype: vec![],
                    obs_records: vec![],
                }
            })
            .collect(),
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn signature(f: &FitResult) -> Vec<u64> {
    std::iter::once(f.ofv)
        .chain(f.theta.iter().copied())
        .chain(f.sigma.iter().copied())
        .chain(f.omega.iter().copied())
        .map(f64::to_bits)
        .chain([f.n_iterations as u64])
        .collect()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let label = args.get(1).map(String::as_str).unwrap_or("unlabelled");
    let rounds = args.get(2).map(|v| v.parse().unwrap()).unwrap_or(7);
    println!("label,case,round,fits,elapsed_ms,signature");
    // All fits have a fixed iteration budget; these are throughput benchmarks,
    // not convergence comparisons. Model/data construction is outside timing.
    for (name, ode, override_ode, width, parallel, count, subjects, iters) in [
        ("short_1t", false, false, Some(1), false, 64, 6, 2),
        ("short_4t", false, false, Some(4), false, 64, 6, 2),
        ("default", false, false, None, false, 32, 24, 5),
        ("batch_analytical", false, false, Some(1), true, 32, 24, 5),
        ("batch_ode", true, true, Some(1), true, 32, 12, 5),
        ("ode_4t", true, true, Some(4), false, 4, 48, 10),
    ] {
        let m = model(ode);
        let p = population(subjects);
        let mut opts = FitOptions {
            threads: width,
            outer_maxiter: iters,
            run_covariance_step: false,
            ..Default::default()
        }
        .quiet();
        if override_ode {
            opts.ode_reltol = 1e-7;
        }
        let expected = signature(&fit(&m, &p, &m.default_params, &opts).unwrap());
        let one = || {
            let f = fit(&m, &p, &m.default_params, &opts).unwrap();
            assert!(f.ofv.is_finite());
            assert_eq!(signature(&f), expected, "{name}: numerical drift");
            if let Some(n) = width {
                assert_eq!(f.n_threads_used, n);
            }
        };
        // Round 0 warms all batch lanes and is reported but excluded from medians.
        for round in 0..=rounds {
            let start = Instant::now();
            if parallel {
                PoolPlan::new(8, 1)
                    .install(|| (0..count).into_par_iter().for_each(|_| one()))
                    .unwrap();
            } else {
                for _ in 0..count {
                    one();
                }
            }
            println!(
                "{label},{name},{round},{count},{:.6},\"{expected:?}\"",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
}
