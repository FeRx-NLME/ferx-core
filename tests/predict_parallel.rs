//! Public `predict()` subject parallelism: ordered, deterministic output and
//! caller-owned nested Rayon budgets.

use ferx_core::{parse_model_file, predict, read_nonmem_csv, PoolPlan, PredictionResult};
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

fn fixture(model_path: &str, copies: usize) -> (ferx_core::CompiledModel, ferx_core::Population) {
    let model = parse_model_file(Path::new(model_path)).expect("model parses");
    let base =
        read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data loads");

    // Enough independent subjects to exercise worker scheduling. Give each
    // copy a unique ID so an ordering regression cannot hide among duplicates.
    let mut subjects = Vec::with_capacity(base.subjects.len() * copies);
    for copy in 0..copies {
        for subject in &base.subjects {
            let mut subject = subject.clone();
            subject.id = format!("{copy}:{}", subject.id);
            subjects.push(subject);
        }
    }
    let mut population = base;
    population.subjects = subjects;
    (model, population)
}

fn assert_same(left: &[PredictionResult], right: &[PredictionResult]) {
    assert_eq!(left.len(), right.len());
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        assert_eq!(left.id, right.id, "ID differs at row {index}");
        assert_eq!(
            left.time.to_bits(),
            right.time.to_bits(),
            "TIME differs at row {index}"
        );
        assert_eq!(
            left.pred.to_bits(),
            right.pred.to_bits(),
            "PRED differs at row {index}"
        );
    }
}

fn assert_deterministic_across_nested_worker_counts(model_path: &str, copies: usize) {
    let (model, population) = fixture(model_path, copies);
    let params = &model.default_params;

    let serial = PoolPlan::new(1, 1)
        .install(|| predict(&model, &population, params).unwrap())
        .expect("one-worker prediction");
    let parallel = PoolPlan::new(4, 1)
        .install(|| predict(&model, &population, params).unwrap())
        .expect("four-worker prediction");
    let standalone = predict(&model, &population, params).unwrap();

    assert_same(&serial, &parallel);
    assert_same(&serial, &standalone);
}

#[test]
fn analytical_predict_is_ordered_and_deterministic_across_worker_counts() {
    assert_deterministic_across_nested_worker_counts("examples/warfarin.ferx", 16);
}

#[test]
fn ode_predict_is_ordered_and_deterministic_across_worker_counts() {
    assert_deterministic_across_nested_worker_counts("examples/warfarin_ode.ferx", 2);
}

fn timed_prediction(
    pool: &PoolPlan,
    model: &ferx_core::CompiledModel,
    population: &ferx_core::Population,
) -> Duration {
    let start = Instant::now();
    pool.install(|| black_box(predict(model, population, &model.default_params).unwrap()))
        .expect("timed prediction");
    start.elapsed()
}

/// Manual optimized-build performance probe. The one-worker run is the serial
/// execution baseline; both runs go through the same public API and persistent
/// pool path. Keep ignored so timing noise never gates CI.
#[test]
#[ignore = "performance probe: run with --release --ignored --nocapture"]
fn report_predict_subject_parallel_speedups() {
    let copies = std::env::var("FERX_PREDICT_BENCH_COPIES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16);
    for (label, path, repetitions) in [
        ("analytical", "examples/warfarin.ferx", 15),
        ("ode", "examples/warfarin_ode.ferx", 7),
    ] {
        let (model, population) = fixture(path, copies);
        let serial_pool = PoolPlan::new(1, 1);
        let parallel_pool = PoolPlan::new(4, 1);
        let serial_value = serial_pool
            .install(|| predict(&model, &population, &model.default_params).unwrap())
            .expect("serial warmup");
        let parallel_value = parallel_pool
            .install(|| predict(&model, &population, &model.default_params).unwrap())
            .expect("parallel warmup");
        assert_same(&serial_value, &parallel_value);

        let mut serial = Vec::with_capacity(repetitions);
        let mut parallel = Vec::with_capacity(repetitions);
        for repetition in 0..repetitions {
            let order = if repetition % 2 == 0 { [1, 4] } else { [4, 1] };
            for workers in order {
                let elapsed = if workers == 1 {
                    timed_prediction(&serial_pool, &model, &population)
                } else {
                    timed_prediction(&parallel_pool, &model, &population)
                };
                if workers == 1 {
                    serial.push(elapsed);
                } else {
                    parallel.push(elapsed);
                }
            }
        }
        serial.sort_unstable();
        parallel.sort_unstable();
        let serial = serial[serial.len() / 2];
        let parallel = parallel[parallel.len() / 2];
        eprintln!(
            "{label} subjects={}: serial={serial:?}, parallel={parallel:?}, speedup={:.3}x",
            population.subjects.len(),
            serial.as_secs_f64() / parallel.as_secs_f64()
        );
    }
}
