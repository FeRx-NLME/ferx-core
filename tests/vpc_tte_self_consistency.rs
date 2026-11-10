//! Tier 2: the Kaplan–Meier VPC band covers data drawn from its own model (#1886, plan T10).
//!
//! The oracle is self-consistency, not a second engine: NONMEM has no Kaplan–Meier
//! VPC, and the `vpc` R package's band depends on its own grid and binning. Each of
//! `M` "observed" datasets is drawn **here**, with this file's own RNG and closed
//! form (exponential event times with log-normal IIV, uniform dropout on `[0, 40]`,
//! an administrative end at 30, about 45 % censored), outside the simulator.
//! `vpc_tte` then simulates `N` replicates of each dataset's design from the same
//! model. If the band and its censoring are right, the replicates are exchangeable
//! with the data and the observed curve falls inside the 5–95 % band at a fixed time
//! in about 90 % of datasets.
//!
//! The design is chosen so the censoring step is visible (measured on macOS arm64,
//! coverage at H/4, H/2, 3H/4 over these 200 datasets): the shipped reverse-KM draw
//! gives 0.925 / 0.895 / 0.910; dropping the per-subject censoring (`Observed` acting
//! as `Horizon`) gives 0.900 / 0.850 / 0.820 and fails, since replicates with no dropout
//! keep more subjects at risk and the late band narrows. Letting event subjects keep
//! `H` (#1886 plan D4) gives 0.910 / 0.870 / 0.855, low but inside the interval at this
//! M; `event_subjects_draw_censoring_from_the_observed_censorings` is the test that
//! kills that one. With lighter dropout (`[0, 60]`, 38 % censored) the `Horizon`
//! mutation stays inside the interval, which is why the dropout is this heavy.

#[cfg(feature = "survival")]
mod self_consistency {
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::{EventType, ObsRecord, Population, Subject};
    use ferx_core::{vpc_tte, SimulateOptions, VpcGrid, VpcTteOptions};
    use rand::{RngExt, SeedableRng};

    const MODEL: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";
    const LAMBDA: f64 = 0.05;
    const OMEGA: f64 = 0.09;
    const H: f64 = 30.0;
    /// Uniform dropout on `[0, DROP_MAX]`.
    const DROP_MAX: f64 = 40.0;
    const N_SUBJ: usize = 100;
    const M: usize = 200;
    const N_SIM: usize = 200;

    /// One dataset: exponential event times with `λ·exp(η)`, censored at
    /// `min(dropout, H)`.
    fn draw_dataset(seed: u64) -> (Population, f64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut n_cens = 0usize;
        let subjects = (1..=N_SUBJ)
            .map(|i| {
                // Box–Muller for η ~ N(0, ω²).
                let (u1, u2): (f64, f64) = (rng.random(), rng.random());
                let z = (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let lambda = LAMBDA * (OMEGA.sqrt() * z).exp();
                let t_event = -(1.0 - rng.random::<f64>()).ln() / lambda;
                let c = (rng.random::<f64>() * DROP_MAX).min(H);
                let (time, event_type) = if t_event <= c {
                    (t_event, EventType::Exact)
                } else {
                    n_cens += 1;
                    (c, EventType::RightCensored)
                };
                Subject {
                    id: i.to_string(),
                    obs_records: vec![ObsRecord::Event {
                        time,
                        event_type,
                        entry_time: 0.0,
                        cmt: 2,
                    }],
                    ..Default::default()
                }
            })
            .collect();
        let pop = Population {
            subjects,
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        (pop, n_cens as f64 / N_SUBJ as f64)
    }

    #[test]
    fn observed_km_coverage_matches_its_expectation() {
        let model = parse_model_string(MODEL).unwrap();
        let grid = vec![H / 4.0, H / 2.0, 3.0 * H / 4.0];
        let mut inside = [0usize; 3];
        let mut cens = 0.0;
        for m in 0..M {
            let (pop, c) = draw_dataset(1_000_000 + m as u64);
            cens += c;
            let opts = VpcTteOptions {
                n_sim: N_SIM,
                grid: VpcGrid::Times(grid.clone()),
                sim: SimulateOptions {
                    seed: Some(m as u64),
                    horizon: Some(H),
                    ..Default::default()
                },
                ..Default::default()
            };
            let v = vpc_tte(&model, &pop, &model.default_params, &opts).unwrap();
            let s = &v.strata[0];
            for k in 0..3 {
                let obs = s.obs_on_grid[k].expect("observed curve defined");
                let (lo, hi) = (s.lo[k].expect("band"), s.hi[k].expect("band"));
                assert!(obs.is_finite() && lo.is_finite() && hi.is_finite());
                if lo <= obs && obs <= hi {
                    inside[k] += 1;
                }
            }
        }
        let cens = cens / M as f64;
        let cov: Vec<f64> = inside.iter().map(|&n| n as f64 / M as f64).collect();
        eprintln!("censored fraction {cens:.3}; coverage at H/4, H/2, 3H/4 = {cov:?}");
        assert!((0.4..0.5).contains(&cens), "dropout design drifted: {cens}");
        for (k, c) in cov.iter().enumerate() {
            assert!(
                (LO..=HI).contains(c),
                "coverage {c} at grid time {} outside [{LO}, {HI}]",
                grid[k]
            );
        }
    }

    /// The 99 % binomial interval for `M = 200` datasets around the exchangeable
    /// expectation `p₀ = 0.90`: `p₀ ± 2.576·√(p₀(1 − p₀)/M)`.
    const LO: f64 = 0.845;
    const HI: f64 = 0.955;
}
