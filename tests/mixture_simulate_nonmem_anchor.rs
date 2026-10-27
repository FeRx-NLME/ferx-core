//! NONMEM anchor for #2163: simulating a `[mixture]` model draws each subject's class
//! from the mixing probabilities and runs it under that class's `MIXNUM`, Ω and Σ.
//!
//! `nonmem_anchor/mixture_sim_class.ctl` simulates the model of
//! `nonmem_anchor/mixture_sim_class.ferx` under `$SIMULATION ONLYSIM` (NONMEM 7.5.1) on
//! 20,000 subjects whose WT, and so whose P(1), runs from 0.14 to 0.76. The two engines
//! draw from different generators, so the comparison is of distributions, not rows:
//! `make_mixture_sim_class.py summary` reduces the NONMEM table to quantiles of IPRED and
//! DV per WT tercile and time (`results/mixture_sim_class_summary.csv`), and this test
//! checks that the share of ferx samples below each NONMEM quantile is that quantile's
//! level. Class 2 clears 4× faster, with a wider Ω and Σ, so the IPRED and DV
//! distributions are bimodal in the class mix and each tercile carries a different mix.
//! Before #2163 every ferx subject ran as class 1.
//!
//! The dataset is a formula (`make_mixture_sim_class.py`), rebuilt here in memory.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::types::Population;
use ferx_core::{read_nonmem_csv, simulate_with_seed, SimOutcome};
use std::io::Write as _;

const N: usize = 20_000;
const TIMES: [f64; 5] = [0.5, 1.0, 2.0, 4.0, 8.0];

fn anchor(name: &str) -> String {
    format!("{}/nonmem_anchor/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// Subject `i` (1-based): WT = 50 + 0.003·(i − 0.5), written to 4 decimals as the
/// generator writes it; a dose of 100 at t = 0 and observations at `TIMES`.
fn population() -> Population {
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,WT\n");
    for i in 1..=N {
        let wt = format!("{:.4}", 50.0 + 0.003 * (i as f64 - 0.5));
        csv.push_str(&format!("{i},0,0,100,1,1,{wt}\n"));
        for t in TIMES {
            csv.push_str(&format!("{i},{t},0,0,0,1,{wt}\n"));
        }
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    read_nonmem_csv(f.path(), Some(&["WT"]), None).unwrap()
}

fn stratum(id: usize) -> usize {
    3 * (id - 1) / N
}

struct Line {
    kind: String,
    stratum: usize,
    time: f64,
    quantity: String,
    p: f64,
    value: f64,
    n: f64,
}

fn summary() -> Vec<Line> {
    let text = std::fs::read_to_string(anchor("results/mixture_sim_class_summary.csv")).unwrap();
    text.lines()
        .skip(1)
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let num = |s: &str| {
                if s.is_empty() {
                    f64::NAN
                } else {
                    s.parse().unwrap()
                }
            };
            Line {
                kind: f[0].to_string(),
                stratum: f[1].parse().unwrap(),
                time: num(f[2]),
                quantity: f[3].to_string(),
                p: num(f[4]),
                value: num(f[5]),
                n: num(f[6]),
            }
        })
        .collect()
}

/// NONMEM drew its classes at the mixing probabilities, so its quantiles are a mixture
/// reference and not, say, all class 1. Measured worst |z|: 1.64 (NONMEM 7.5.1).
#[test]
fn nonmem_draws_classes_at_the_mixing_probabilities() {
    let lines = summary();
    let mut worst = 0.0f64;
    for s in 0..3 {
        let get = |kind: &str| {
            lines
                .iter()
                .find(|l| l.kind == kind && l.stratum == s)
                .unwrap_or_else(|| panic!("{kind} for stratum {s}"))
        };
        let (share, p1) = (get("class1_share"), get("mean_p1"));
        let z = (share.value - p1.value) / (p1.value * (1.0 - p1.value) / share.n).sqrt();
        assert!(z.is_finite(), "stratum {s}: z = {z}");
        assert!(
            z.abs() < 4.0,
            "stratum {s}: NONMEM class-1 share {} against P(1) {}, z = {z}",
            share.value,
            p1.value
        );
        worst = worst.max(z.abs());
    }
    eprintln!("worst class-share |z| = {worst:.3}");
}

/// The share of ferx samples below each NONMEM quantile Q_p is p, within 4 SD of the
/// two-sample difference, `sqrt(2·p(1−p)/n)`: NONMEM's quantile carries the sampling
/// error of its own n. 150 comparisons (3 terciles × 5 times × IPRED/DV × 5 levels).
/// Measured worst |z|: 2.42 (DV at t = 0.5, stratum 1, p = 0.9); the largest of 150
/// independent N(0, 1) draws is about 2.7. All-class-1 ferx (before #2163): 93.2.
#[test]
fn ferx_simulates_the_nonmem_mixture_distribution() {
    let model = parse_model_file(std::path::Path::new(&anchor("mixture_sim_class.ferx"))).unwrap();
    let rows = simulate_with_seed(&model, &population(), &model.default_params, 1, 2163).unwrap();
    assert_eq!(rows.len(), N * TIMES.len());

    let mut worst = (0.0f64, String::new());
    let mut compared = 0;
    for l in summary().iter().filter(|l| l.kind == "quantile") {
        let values: Vec<f64> = rows
            .iter()
            .filter(|r| stratum(r.id.parse().unwrap()) == l.stratum && r.time == l.time)
            .map(|r| match (l.quantity.as_str(), &r.outcome) {
                ("IPRED", _) => r.ipred,
                ("DV", SimOutcome::Continuous { value }) => *value,
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(
            values.len() as f64,
            l.n,
            "stratum {} at t = {}",
            l.stratum,
            l.time
        );
        assert!(
            values.iter().all(|v| v.is_finite()),
            "a non-finite ferx sample"
        );
        let below = values.iter().filter(|&&v| v < l.value).count() as f64 / l.n;
        let z = (below - l.p) / (2.0 * l.p * (1.0 - l.p) / l.n).sqrt();
        let at = format!(
            "{} at t = {}, stratum {}, p = {}: ferx {below:.4} below NONMEM's {}",
            l.quantity, l.time, l.stratum, l.p, l.value
        );
        assert!(z.is_finite(), "{at} (z = {z})");
        if z.abs() > worst.0 {
            worst = (z.abs(), at);
        }
        compared += 1;
    }
    assert_eq!(compared, 150);
    eprintln!("worst |z| = {:.3}: {}", worst.0, worst.1);
    assert!(worst.0 < 4.0, "{} (|z| = {:.2})", worst.1, worst.0);
}
