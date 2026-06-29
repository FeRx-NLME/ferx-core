//! Generate **ferx-simulated** joint PK-TTE event times for the Slice 2.2 cross-tool
//! simulation anchor (`tests/reference/pktte_joint_sim/`). This is the ferx counterpart
//! of that directory's `sim.ctl` (NONMEM `$SIM`): both simulate the *same* design and
//! `compare.R` checks the event-time distributions agree (ferx's KM vs NONMEM's analytic
//! marginal survival). The sampler's *exactness* is pinned separately and rigorously by
//! the PIT/KS goodness-of-fit unit test (`joint_pktte_event_times_match_model_survival`);
//! this anchor is the external cross-tool corroboration.
//!
//! Design (matches `make_template.py` / `sim.ctl`): oral 1-cpt PK + drug-driven hazard
//! `h = H0·exp(BETA·Cc)` accumulated as an ODE state, single dose=100, horizon=24, BSV on
//! CL. Truth: CL=1, V=10, KA=1, H0=0.015, BETA=0.25, ω²(CL)=0.09.
//!
//! Run (dev tooling, like `generate_data.rs` — gated on `survival`):
//!   cargo run --release --bin pktte_sim_anchor --features survival -- ferx_events.csv
//! Writes `ID,TIME,DV` (DV=1 observed event, 0 right-censored at the horizon).

#[cfg(feature = "survival")]
fn main() {
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::{DoseEvent, EventType, ObsRecord, Population};
    use ferx_core::{simulate_with_options, SimOutcome, SimulateOptions};

    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ferx_events.csv".into());
    const N: usize = 500;
    const DOSE: f64 = 100.0;
    const HORIZON: f64 = 24.0;
    const SEED: u64 = 20260629;

    // Same truth as sim.ctl (and ../pktte_joint/simulate.R).
    let model_src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.015, 1e-5, 10.0)
  theta TVBETA(0.25, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[event_model]
  cmt    = 3
  hazard = H0 * exp(BETA * (central / V))

[error_model]
  DV ~ proportional(PROP_ERR)
";
    let model = parse_model_string(model_src).expect("anchor model must parse");

    // One depot dose + a TTE placeholder right-censored at the horizon (CMT 3); no
    // continuous PK observations (the event sampler is what we are exporting).
    use std::collections::HashMap;
    let subjects = (0..N)
        .map(|i| ferx_core::types::Subject {
            id: (i + 1).to_string(),
            doses: vec![DoseEvent::new(0.0, DOSE, 1, 0.0, false, 0.0)],
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
                cmt: 3,
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
        seed: Some(SEED),
        match_method: None,
        horizon: Some(HORIZON),
    };
    let sims = simulate_with_options(&model, &pop, &model.default_params, 1, &opts)
        .expect("ferx joint PK-TTE simulation must succeed");

    let mut csv = String::from("ID,TIME,DV\n");
    let (mut ev, mut cens) = (0usize, 0usize);
    for r in &sims {
        if let SimOutcome::Event { time, observed } = r.outcome {
            csv.push_str(&format!("{},{:.6},{}\n", r.id, time, observed as u8));
            if observed {
                ev += 1;
            } else {
                cens += 1;
            }
        }
    }
    std::fs::write(&out, csv).expect("write ferx_events.csv");
    eprintln!(
        "ferx: wrote {out} — N={N} events={ev} censored={cens} ({:.0}% events)",
        100.0 * ev as f64 / N as f64
    );
}

#[cfg(not(feature = "survival"))]
fn main() {
    eprintln!("build with --features survival");
    std::process::exit(1);
}
