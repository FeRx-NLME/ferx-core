//! Regenerates the peak-memory probe's dataset (#3081), from the repo root:
//! `cargo run --release --example generate_peak_memory`
//!
//! Writes `tools/peak-memory/stiff_pkpd.csv`: 100 subjects simulated from
//! `tools/peak-memory/stiff_pkpd.ferx` at its initial estimates (seed 3081). Two 21-day
//! cycles of a 5-day daily oral dose (10 dose segments per subject), PK sampled on CMT 2
//! and the circulating-cell PD on CMT 9 — the multi-dose, two-endpoint shape of the
//! private e7070 model the probe stands in for. Dev tooling, kept out of `src/` because
//! it reads and writes repository paths.

use ferx_core::*;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

const MODEL: &str = "tools/peak-memory/stiff_pkpd.ferx";
const OUT: &str = "tools/peak-memory/stiff_pkpd.csv";
const N_SUBJECTS: usize = 100;
const DOSE: f64 = 300.0;

fn main() {
    let dose_times: Vec<f64> =
        [0.0, 504.0].iter().flat_map(|c| (0..5).map(move |d| c + 24.0 * d as f64)).collect();
    let pk_times = [0.5, 1.0, 2.0, 4.0, 8.0, 23.5, 96.5, 98.0, 100.0, 104.0, 505.0, 508.0];
    let pd_times = [0.01, 168.0, 240.0, 336.0, 420.0, 503.5, 672.0, 840.0, 1008.0];

    // (time, evid, cmt) rows of one subject, time-sorted; no observation shares a dose time.
    let mut rows: Vec<(f64, u8, usize)> = dose_times.iter().map(|&t| (t, 1, 1)).collect();
    rows.extend(pk_times.iter().map(|&t| (t, 0, 2)));
    rows.extend(pd_times.iter().map(|&t| (t, 0, 9)));
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));

    let write = |path: &Path, dv: &dyn Fn(usize, usize) -> f64| {
        let mut f = std::fs::File::create(path).unwrap();
        writeln!(f, "ID,TIME,DV,EVID,AMT,CMT,MDV").unwrap();
        for id in 1..=N_SUBJECTS {
            let mut k = 0;
            for &(t, evid, cmt) in &rows {
                if evid == 1 {
                    writeln!(f, "{id},{t},.,1,{DOSE},{cmt},1").unwrap();
                } else {
                    writeln!(f, "{id},{t},{:.6},0,.,{cmt},0", dv(id, k)).unwrap();
                    k += 1;
                }
            }
        }
    };

    // The design with placeholder DVs, read back through the ordinary reader so the
    // simulated subjects are exactly what `fit()` will see.
    let design = std::env::temp_dir().join("ferx-peak-memory-design.csv");
    write(&design, &|_, _| 1.0);
    let pop = read_nonmem_csv(&design, None, None).unwrap();
    let _ = std::fs::remove_file(&design);

    let model = parse_model_file(Path::new(MODEL)).unwrap();
    let sim = simulate_with_seed(&model, &pop, &model.default_params, 1, 3081).unwrap();
    let mut dv: HashMap<(String, usize), f64> = HashMap::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for s in &sim {
        let k = seen.entry(s.id.clone()).or_insert(0);
        dv.insert((s.id.clone(), *k), s.outcome.continuous_value());
        *k += 1;
    }
    write(Path::new(OUT), &|id, k| dv[&(id.to_string(), k)]);
    eprintln!("Written: {OUT}");
}
