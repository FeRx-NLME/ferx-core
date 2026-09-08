//! Usage: focei_profile CASE THREADS REPEATS MODE OUTPUT_JSON
//! CASE: analytical | iov | stiff. MODE: time | kernels | counts | stacks | byte_stacks.
//! Allocation builds: cargo rustc --profile ci-test --example focei_profile --
//! --cfg profiling_allocations -C debuginfo=1. Normal builds have no allocator hook.
#![allow(unexpected_cfgs)] // example-only --cfg, not an engine feature/API
use ferx_core::estimation::{
    inner_optimizer as inner,
    parameterization::{pack_params, unpack_params},
    sens_outer_gradient as gradient,
};
use ferx_core::sens::provider;
use ferx_core::{fit, parser::model_parser::parse_model_string, read_nonmem_csv, types::*};
use serde_json::{json, Value};
use std::{hint::black_box, path::Path, sync::atomic::Ordering::Relaxed, time::Instant};

#[cfg(profiling_allocations)]
#[path = "../tools/profiling_alloc.rs"]
mod allocation;
#[cfg(profiling_allocations)]
#[global_allocator]
static ALLOCATOR: allocation::Allocator = allocation::Allocator;

fn counters() -> Value {
    json!({"full_sens_calls": provider::PROFILE_SENS_CALLS.load(Relaxed),
        "full_sens_ns": provider::PROFILE_SENS_NANOS.load(Relaxed),
        "eta_sens_calls": provider::PROFILE_ETA_CALLS.load(Relaxed),
        "eta_sens_ns": provider::PROFILE_ETA_NANOS.load(Relaxed),
        "inner_solves": inner::PROFILE_INNER_SOLVES.load(Relaxed),
        "inner_analytic_steps": inner::PROFILE_INNER_ANALYTIC_GRAD.load(Relaxed),
        "inner_fd_steps": inner::PROFILE_INNER_FD_FALLBACK.load(Relaxed)})
}
fn reset_counters() {
    for c in [
        &provider::PROFILE_SENS_CALLS,
        &provider::PROFILE_SENS_NANOS,
        &provider::PROFILE_ETA_CALLS,
        &provider::PROFILE_ETA_NANOS,
        &inner::PROFILE_INNER_SOLVES,
        &inner::PROFILE_INNER_ANALYTIC_GRAD,
        &inner::PROFILE_INNER_FD_FALLBACK,
    ] {
        c.store(0, Relaxed);
    }
}
fn measure<T>(name: &str, stride: u64, op: impl FnOnce() -> T) -> (T, Value) {
    reset_counters();
    #[cfg(profiling_allocations)]
    allocation::start(stride);
    #[cfg(not(profiling_allocations))]
    let _ = stride;
    let start = Instant::now();
    let result = op();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    #[cfg(profiling_allocations)]
    let allocations = allocation::stop();
    #[cfg(not(profiling_allocations))]
    let allocations = Value::Null;
    (
        result,
        json!({"phase":name,"elapsed_ms":elapsed,"providers":counters(),"allocations":allocations}),
    )
}

fn exact_binding(t: f64, cl: f64, v: f64) -> f64 {
    let a = -cl / v - 1e4;
    let d = -1e3;
    let trace = a + d;
    let det = cl / v * 1e3;
    let big = (trace - (trace * trace - 4.0 * det).sqrt()) / 2.0;
    let small = det / big;
    100.0 / v * ((small * t).exp() * (a - big) - (big * t).exp() * (a - small)) / (small - big)
}
fn fixture(case: &str) -> (CompiledModel, Population) {
    if case != "stiff" {
        let (model_file, data_file, occ) = match case {
            "analytical" => ("examples/warfarin.ferx", "data/warfarin.csv", None),
            "iov" => (
                "examples/warfarin_iov.ferx",
                "data/warfarin_iov.csv",
                Some("OCC"),
            ),
            _ => panic!("unknown case: {case}"),
        };
        let model = parse_model_string(&std::fs::read_to_string(model_file).unwrap()).unwrap();
        return (
            model,
            read_nonmem_csv(Path::new(data_file), None, occ).unwrap(),
        );
    }
    let model = parse_model_string(
        r"
[parameters]
 theta TVCL(3, 0.1, 30)
 theta TVV(20, 2, 200)
 omega ETA_CL ~ 0.04
 sigma PROP ~ 0.01
[individual_parameters]
 CL = TVCL * exp(ETA_CL)
 V = TVV
 KON = 10000
 KOFF = 1000
[structural_model]
 ode(states=[central,bound])
[odes]
 d/dt(central) = -(CL/V)*central - KON*central + KOFF*bound
 d/dt(bound) = KON*central - KOFF*bound
[scaling]
 y = central/V
[error_model]
 DV ~ proportional(PROP)
[fit_options]
 ode_method = rodas5p
 ode_reltol = 1e-6
 ode_abstol = 1e-8
",
    )
    .unwrap();
    let mut pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).unwrap();
    pop.subjects.truncate(4);
    for (i, s) in pop.subjects.iter_mut().enumerate() {
        s.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
        s.obs_times = vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
        s.obs_raw_times.clear();
        let cl = 3.0 * (0.08 * (i as f64 - 1.5)).exp();
        s.observations = s
            .obs_times
            .iter()
            .enumerate()
            .map(|(j, &t)| exact_binding(t, cl, 20.0) * (0.95 + 0.02 * ((i + j) % 6) as f64))
            .collect();
        s.obs_cmts = vec![1; 8];
        s.cens = vec![0; 8];
    }
    // Check the stiff trajectory against its matrix-exponential closed form.
    for pred in ferx_core::predict(&model, &pop, &model.default_params) {
        let exact = exact_binding(pred.time, 3.0, 20.0);
        assert!((pred.pred - exact).abs() < 2e-5 * exact.abs().max(1e-6));
    }
    (model, pop)
}
fn signature(f: &FitResult) -> Vec<u64> {
    std::iter::once(f.ofv)
        .chain(f.theta.iter().copied())
        .chain(f.sigma.iter().copied())
        .chain(f.omega.iter().copied())
        .chain(f.omega_iov.iter().flat_map(|m| m.iter().copied()))
        .map(f64::to_bits)
        .chain([f.n_iterations as u64])
        .collect()
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let case = &args[1];
    let threads: usize = args[2].parse().unwrap();
    let repeats: usize = args[3].parse().unwrap();
    let mode = &args[4];
    assert!(repeats > 0);
    assert!(["time", "kernels", "counts", "stacks", "byte_stacks"].contains(&mode.as_str()));
    assert_eq!(
        mode == "counts" || mode == "stacks" || mode == "byte_stacks",
        cfg!(profiling_allocations),
        "counts/stacks need the allocation build; time/kernels need the normal build"
    );
    #[cfg(profiling_allocations)]
    allocation::sample_bytes(mode == "byte_stacks");
    let (model, pop) = fixture(case);
    let params = &model.default_params;
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        threads: Some(threads),
        outer_maxiter: 5,
        inner_maxiter: 30,
        run_covariance_step: false,
        ..Default::default()
    }
    .quiet();
    let reference = fit(&model, &pop, params, &opts).expect("reference fit");
    assert!(reference.ofv.is_finite());
    let expected = signature(&reference);
    let mut measurements = Vec::new();
    let mut stride = 0;
    for _ in 0..repeats {
        let (result, row) = measure("full_fit", stride, || {
            fit(&model, &pop, params, &opts).unwrap()
        });
        assert_eq!(
            signature(&result),
            expected,
            "repeated fit changed numerically"
        );
        assert_eq!(result.n_threads_used, threads);
        if mode == "stacks" || mode == "byte_stacks" {
            let metric = if mode == "byte_stacks" {
                "requested_bytes"
            } else {
                "allocation_calls"
            };
            stride = (row["allocations"][metric].as_u64().unwrap() / 128).max(1);
        }
        measurements.push(row);
    }
    // Pointwise kernel breakdown, separate from the full optimizer trajectory.
    if mode != "time" {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(ferx_core::FIT_RAYON_STACK_SIZE)
            .build()
            .unwrap();
        pool.install(|| {
            let (ebes, row) = measure("inner_cold_at_initial", 0, || {
                inner::run_inner_loop_warm(
                    &model,
                    &pop,
                    params,
                    opts.inner_maxiter,
                    opts.inner_tol,
                    None,
                    None,
                    0,
                    0,
                )
            });
            measurements.push(row);
            let (_, row) = measure("inner_warm_same_point", 0, || {
                inner::run_inner_loop_warm(
                    &model,
                    &pop,
                    params,
                    opts.inner_maxiter,
                    opts.inner_tol,
                    Some(&ebes.0),
                    None,
                    0,
                    0,
                )
            });
            measurements.push(row);
            let (value, row) = measure("likelihood_at_initial", 0, || {
                if let Some(iov) = &params.omega_iov {
                    ferx_core::stats::likelihood::foce_population_nll_iov(
                        &model,
                        &pop,
                        &params.theta,
                        &ebes.0,
                        &ebes.1,
                        &ebes.3,
                        &params.omega,
                        iov,
                        &params.sigma.values,
                        true,
                    )
                } else {
                    ferx_core::stats::likelihood::foce_population_nll(
                        &model,
                        &pop,
                        &params.theta,
                        &ebes.0,
                        &ebes.1,
                        &params.omega,
                        &params.sigma.values,
                        &params.residual_correlations,
                        true,
                    )
                }
            });
            assert!(value.is_finite());
            measurements.push(row);
            let x = pack_params(params);
            let (grads, mut row) = measure("analytic_gradient_at_initial", 0, || {
                if model.n_kappa > 0 {
                    gradient::per_subject_packed_gradients_iov(
                        &model, &pop, params, &x, &ebes.0, &ebes.3, true,
                    )
                } else {
                    gradient::per_subject_packed_gradients(
                        &model, &pop, params, &x, &ebes.0, true, None,
                    )
                }
            });
            row["unsupported_subjects"] = json!(grads.iter().filter(|g| g.is_none()).count());
            measurements.push(row);
            let (_, row) = measure("unpack_params_1000_calls", 0, || {
                for _ in 0..1000 {
                    black_box(unpack_params(black_box(&x), params));
                }
            });
            measurements.push(row);
        });
    }
    let report = json!({"case":case,"threads":threads,"mode":mode,
        "allocation_build":cfg!(profiling_allocations),"signature":expected,
        "subjects":pop.subjects.len(),"observations":pop.subjects.iter().map(|s|s.observations.len()).sum::<usize>(),
        "theta":model.n_theta,"eta":model.n_eta,"kappa":model.n_kappa,
        "outer_maxiter":opts.outer_maxiter,"inner_maxiter":opts.inner_maxiter,"measurements":measurements});
    std::fs::write(&args[5], serde_json::to_string_pretty(&report).unwrap()).unwrap();
    println!("{case} threads={threads} mode={mode} -> {}", args[5]);
}
