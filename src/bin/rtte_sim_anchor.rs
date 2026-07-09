//! Generate **ferx-simulated** repeated-TTE (RTTE) data for the Slice 3.3 cross-tool
//! simulation anchor (`tests/reference/rtte_exponential_sim/`). ferx simulates a
//! clock-forward exponential RTTE dataset from a known `(λ, ω²)`; the same file is then
//! fitted by **both** ferx and NONMEM (`nonmem.ctl`, the Slice 3.1 telescoping-AG
//! control stream pointed at this dataset), and both must recover the data-generating
//! `(λ, ω²)`. Fitting ferx-simulated data with an *independent* engine (NONMEM) is the
//! external corroboration that the RTTE **simulator** is correct — a biased sampler
//! would shift the recovered parameters away from the truth in both engines.
//!
//! The sampler's *exactness* is pinned separately by the estimator-free method-of-
//! moments and PIT/KS checks in `tte_convergence.rs`; this anchor is the cross-tool leg.
//!
//! Truth (matches `tests/reference/rtte_exponential`): TVLAMBDA = 0.15, ω²(log rate) =
//! 0.09, horizon = 20 → ~3 events/subject.
//!
//! Run (dev tooling, like `generate_data.rs` / `pktte_sim_anchor.rs` — gated on
//! `survival`; excluded from coverage in `codecov.yml`):
//!   cargo run --release --bin rtte_sim_anchor --features survival -- \
//!       tests/reference/rtte_exponential_sim/rtte_sim.csv
//! Writes NONMEM-format `ID,TIME,DV,EVID,CMT,MDV` (DV=1 event, DV=0 admin censor at the
//! horizon), one recurrent stream per subject, time-sorted, ending in a censor row.

#[cfg(feature = "survival")]
fn main() {
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::{EventType, ObsRecord, Population};
    use ferx_core::{simulate_with_options, SimOutcome, SimulateOptions};
    use std::collections::BTreeMap;

    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "rtte_sim.csv".into());
    const N: usize = 300;
    const HORIZON: f64 = 20.0;
    const SEED: u64 = 20260709;

    // Same truth as tests/reference/rtte_exponential (clock-forward constant hazard,
    // shared log-rate frailty).
    let model_src = r"
[parameters]
  theta TVLAMBDA(0.15, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = 2
  type   = rtte
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";
    let model = parse_model_string(model_src).expect("RTTE anchor model must parse");

    // One right-censored template row per subject on CMT 2 at the horizon;
    // `simulate_rtte_stream` regenerates each subject's recurrent stream.
    let template = {
        use ferx_core::types::Subject;
        use std::collections::HashMap;
        let subjects = (0..N)
            .map(|i| Subject {
                id: (i + 1).to_string(),
                doses: vec![],
                obs_times: vec![],
                obs_raw_times: vec![],
                observations: vec![],
                obs_cmts: vec![],
                covariates: HashMap::new(),
                dose_covariates: vec![],
                obs_covariates: vec![],
                pk_only_times: vec![],
                pk_only_covariates: vec![],
                reset_times: vec![],
                cens: vec![],
                occasions: vec![],
                dose_occasions: vec![],
                fremtype: vec![],
                obs_records: vec![ObsRecord::Event {
                    time: HORIZON,
                    event_type: EventType::RightCensored,
                    entry_time: 0.0,
                    cmt: 2,
                }],
            })
            .collect();
        Population {
            subjects,
            covariate_names: vec![],
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    };

    let opts = SimulateOptions {
        seed: Some(SEED),
        match_method: None,
        horizon: Some(HORIZON),
    };
    let sims = simulate_with_options(&model, &template, &model.default_params, 1, &opts)
        .expect("ferx RTTE simulation must succeed");

    // Group rows by subject id (numeric order) so the CSV is per-subject, time-sorted.
    let mut by_id: BTreeMap<u32, Vec<(f64, u8)>> = BTreeMap::new();
    let (mut n_events, mut n_subj_with_event) = (0usize, 0usize);
    for r in &sims {
        if let SimOutcome::Event { time, observed } = r.outcome {
            let id: u32 = r.id.parse().expect("numeric subject id");
            by_id.entry(id).or_default().push((time, observed as u8));
        }
    }
    let mut csv = String::from("ID,TIME,DV,EVID,CMT,MDV\n");
    for (id, rows) in &by_id {
        let events_here = rows.iter().filter(|(_, dv)| *dv == 1).count();
        n_events += events_here;
        if events_here > 0 {
            n_subj_with_event += 1;
        }
        for (time, dv) in rows {
            csv.push_str(&format!("{id},{time:.4},{dv},0,2,0\n"));
        }
    }
    std::fs::write(&out, csv).expect("write rtte_sim.csv");
    eprintln!(
        "ferx: wrote {out} — N={N} subjects, {n_events} events ({:.2}/subject), \
         {n_subj_with_event} subjects with ≥1 event",
        n_events as f64 / N as f64
    );

    // ── Diagnostic: does the sampler inject the correct frailty variance? ──
    // The population log-rate variance is ω² = 0.09 by construction. The
    // method-of-moments estimate from the counts, ω²_MoM = ln(1 + (Var−μ)/μ²),
    // is finite-sample biased HIGH when events/subject is small (~3) — the count
    // over-dispersion is a noisy read on the frailty there. As the horizon (hence
    // events/subject) grows, each subject's rate is pinned precisely and MoM → 0.09.
    // If it did NOT converge, that would flag a genuine over-dispersion bug.
    let mom = |n_subj: usize, horizon: f64, seed: u64| -> (f64, f64, f64) {
        use ferx_core::types::Subject;
        use std::collections::HashMap;
        let subjects = (0..n_subj)
            .map(|i| Subject {
                id: (i + 1).to_string(),
                doses: vec![],
                obs_times: vec![],
                obs_raw_times: vec![],
                observations: vec![],
                obs_cmts: vec![],
                covariates: HashMap::new(),
                dose_covariates: vec![],
                obs_covariates: vec![],
                pk_only_times: vec![],
                pk_only_covariates: vec![],
                reset_times: vec![],
                cens: vec![],
                occasions: vec![],
                dose_occasions: vec![],
                fremtype: vec![],
                obs_records: vec![ObsRecord::Event {
                    time: horizon,
                    event_type: EventType::RightCensored,
                    entry_time: 0.0,
                    cmt: 2,
                }],
            })
            .collect();
        let pop = Population {
            subjects,
            covariate_names: vec![],
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let opts = SimulateOptions {
            seed: Some(seed),
            match_method: None,
            horizon: Some(horizon),
        };
        let sims = simulate_with_options(&model, &pop, &model.default_params, 1, &opts).unwrap();
        let mut by = BTreeMap::<String, usize>::new();
        for r in &sims {
            let c = by.entry(r.id.clone()).or_default();
            if matches!(r.outcome, SimOutcome::Event { observed: true, .. }) {
                *c += 1;
            }
        }
        let counts: Vec<f64> = by.values().map(|&c| c as f64).collect();
        let m = counts.iter().sum::<f64>() / counts.len() as f64;
        let v = counts.iter().map(|c| (c - m).powi(2)).sum::<f64>() / (counts.len() as f64 - 1.0);
        (m, v, (1.0 + (v - m) / (m * m)).ln())
    };
    eprintln!("\nfrailty-variance (MoM ω², truth 0.09) vs events/subject:");
    for (h, seed) in [(20.0, 1), (60.0, 2), (200.0, 3), (600.0, 4)] {
        let (m, v, w2) = mom(4000, h, seed);
        eprintln!("  horizon={h:>4}  mean={m:6.2}  var={v:7.2}  MoM ω²={w2:.4}");
    }
}

#[cfg(not(feature = "survival"))]
fn main() {
    eprintln!("build with --features survival");
    std::process::exit(1);
}
