//! Tier 2: the repeated-events VPC bands cover data drawn from their own model (#1886
//! PR2, plan T13).
//!
//! The oracle is self-consistency, not a second engine: NONMEM has no Kaplan–Meier or
//! mean-cumulative-function VPC. Each of `M` "observed" datasets is drawn **here**, with
//! this file's own RNG and closed form, outside the simulator: a Poisson process of
//! rate `λ·exp(η)` per subject, followed to a uniform dropout time on `[0, 25]` capped
//! at the administrative end `H = 20`, so about 80 % of subjects leave before `H`.
//! `vpc_rtte` then simulates `N` replicates of each dataset's design from the same
//! model. If the bands and the follow-up of the replicates are right, the replicates
//! are exchangeable with the data and an observed curve falls inside its 5–95 % band at
//! a fixed time in about 90 % of datasets.
//!
//! Two curves are checked at H/4, H/2 and 3H/4: the mean cumulative function and the
//! order-2 Kaplan–Meier curve on absolute time. The design is chosen so the follow-up
//! step is visible. Measured on macOS arm64 over these 200 datasets (MCF, then order-2
//! KM, each at H/4, H/2, 3H/4):
//!
//! - as shipped: 0.890 / 0.890 / 0.920 and 0.890 / 0.850 / 0.925;
//! - `Observed` acting as `Horizon` (every replicate subject followed to `H`):
//!   0.885 / 0.845 / 0.835 and 0.870 / 0.790 / 0.820, which fails at three points.
//!   Replicates with no dropout keep more subjects at risk and the band narrows.
//!
//! The shipped coverage is a little under 0.90 on average and the margin to the lower
//! bound is small: two other seed sets gave a lowest point of 0.840 and 0.855 (pooled
//! over the three sets, 600 datasets, every point is between 0.873 and 0.895). A band
//! read from 200 replicates covers slightly less than its nominal 90 %. So this test
//! pins these seeds, and a change that moves the draws can move a point across the
//! bound without being a defect: re-measure the pooled coverage before reading a
//! failure here as one.

#[cfg(feature = "survival")]
mod self_consistency {
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::{EventType, ObsRecord, Population, Subject};
    use ferx_core::{vpc_rtte, SimulateOptions, VpcGrid, VpcRtteOptions};
    use rand::{RngExt, SeedableRng};

    const MODEL: &str = r"
[parameters]
  theta TVLAMBDA(0.15, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = 2
  type   = rtte
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";
    const LAMBDA: f64 = 0.15;
    const OMEGA: f64 = 0.09;
    const H: f64 = 20.0;
    /// Uniform dropout on `[0, DROP_MAX]`, capped at `H`.
    const DROP_MAX: f64 = 25.0;
    const N_SUBJ: usize = 100;
    const M: usize = 200;
    const N_SIM: usize = 200;

    /// One dataset: per subject a Poisson process of rate `λ·exp(η)` on `[0, c]` and a
    /// closing censoring record at `c = min(dropout, H)`. Returns the fraction of
    /// subjects that left before `H`.
    fn draw_dataset(seed: u64) -> (Population, f64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut n_early = 0usize;
        let subjects = (1..=N_SUBJ)
            .map(|i| {
                // Box–Muller for η ~ N(0, ω²).
                let (u1, u2): (f64, f64) = (rng.random(), rng.random());
                let z = (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let rate = LAMBDA * (OMEGA.sqrt() * z).exp();
                let c = (rng.random::<f64>() * DROP_MAX).min(H);
                if c < H {
                    n_early += 1;
                }
                let record = |time, event_type| ObsRecord::Event {
                    time,
                    event_type,
                    entry_time: 0.0,
                    cmt: 2,
                };
                let mut obs_records = Vec::new();
                let mut t = 0.0;
                loop {
                    t += -(1.0 - rng.random::<f64>()).ln() / rate;
                    if t >= c {
                        break;
                    }
                    obs_records.push(record(t, EventType::Exact));
                }
                obs_records.push(record(c, EventType::RightCensored));
                Subject {
                    id: i.to_string(),
                    obs_records,
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
        (pop, n_early as f64 / N_SUBJ as f64)
    }

    /// T13. The measured coverage and the mutation that fails it are in the file doc.
    #[test]
    fn observed_rtte_curves_coverage_matches_its_expectation() {
        let model = parse_model_string(MODEL).unwrap();
        let grid = vec![H / 4.0, H / 2.0, 3.0 * H / 4.0];
        // inside[curve][grid point]: curve 0 is the MCF, 1 the order-2 absolute KM.
        let mut inside = [[0usize; 3]; 2];
        let mut early = 0.0;
        for m in 0..M {
            let (pop, e) = draw_dataset(2_000_000 + m as u64);
            early += e;
            let opts = VpcRtteOptions {
                n_sim: N_SIM,
                grid: VpcGrid::Times(grid.clone()),
                max_order: Some(2),
                sim: SimulateOptions {
                    seed: Some(m as u64),
                    horizon: Some(H),
                    ..Default::default()
                },
                ..Default::default()
            };
            let v = vpc_rtte(&model, &pop, &model.default_params, &opts).unwrap();
            let s = &v.strata[0];
            let km2 = &s.absolute[1];
            assert_eq!(km2.order, 2);
            let curves = [
                (&s.mcf.obs_on_grid, &s.mcf.lo, &s.mcf.hi),
                (&km2.obs_on_grid, &km2.lo, &km2.hi),
            ];
            for (j, (obs, lo, hi)) in curves.into_iter().enumerate() {
                for k in 0..3 {
                    let obs = obs[k].expect("observed curve defined");
                    let (lo, hi) = (lo[k].expect("band"), hi[k].expect("band"));
                    assert!(obs.is_finite() && lo.is_finite() && hi.is_finite());
                    if lo <= obs && obs <= hi {
                        inside[j][k] += 1;
                    }
                }
            }
        }
        let early = early / M as f64;
        let cov: Vec<Vec<f64>> = inside
            .iter()
            .map(|c| c.iter().map(|&n| n as f64 / M as f64).collect())
            .collect();
        eprintln!(
            "left before H: {early:.3}; coverage at H/4, H/2, 3H/4: MCF {:?}, order-2 KM {:?}",
            cov[0], cov[1]
        );
        assert!(
            (0.75..0.85).contains(&early),
            "dropout design drifted: {early}"
        );
        for (name, c) in ["MCF", "order-2 KM"].iter().zip(&cov) {
            for (k, c) in c.iter().enumerate() {
                assert!(
                    (LO..=HI).contains(c),
                    "{name}: coverage {c} at grid time {} outside [{LO}, {HI}]",
                    grid[k]
                );
            }
        }
    }

    /// The 99 % binomial interval for `M = 200` datasets around the exchangeable
    /// expectation 0.90: `qbinom(c(.005, .995), 200, .9) / 200` in R.
    const LO: f64 = 0.84;
    const HI: f64 = 0.95;
}
