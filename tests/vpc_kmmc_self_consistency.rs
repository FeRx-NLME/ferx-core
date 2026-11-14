//! Tier 2: the Kaplan–Meier mean covariate VPC covers data drawn from its own model
//! (K9), and detects a model that lacks the covariate effect (K10) (#1886 PR3).
//!
//! The oracle is self-consistency, not a second engine: NONMEM has no Kaplan–Meier
//! estimator, and the reference implementations disagree on the risk set (Xpose 4)
//! or mix strata (vpc 1.2.4). Each of `M` "observed" datasets is drawn **here**, with
//! this file's own RNG and closed form, outside the simulator: `Z ~ N(0, 1)`,
//! exponential event times with hazard `0.05·exp(0.7·Z)`, uniform dropout on
//! `[0, 30]` and an administrative end at `H = 30`. `vpc_kmmc` then simulates `N_SIM`
//! replicates of each dataset's design. If the band and its censoring are right, the
//! observed curve falls inside the 5–95 % band at a fixed time in about 89 % of
//! datasets (`0.90·(N − 1)/(N + 1) = 0.891` for a type-7 band from 200 replicates).
//! The floor of 10 at risk applies to each curve on its own; a dataset counts at a
//! time only where its observed curve and its band are both defined.
//!
//! The R prototype of this design (`~/ferx-probes/1886/pr3/coverage.R`, plan M6/M7,
//! a different RNG) gave 0.905 / 0.900 / 0.900 at H/4, 3H/8, H/2 for the own model,
//! and 0.185 / 0.170 / 0.255 for the no-covariate model while the KM-VPC covered
//! 0.815 / 0.785 / 0.66.
//!
//! Measured in ferx on macOS arm64 at the commit that added this file, over these 200
//! datasets: own model 0.905 / 0.920 / 0.915; no-covariate model outside the band in
//! 0.82 / 0.815 / 0.79 of datasets, while its KM-VPC covers 0.84 / 0.70 / 0.575; both
//! tests together 1.5 s under `--profile ci-test`. Mutations (each run on both tests):
//! censoring replicates at `H` only takes K9 to 0.765 / 0.68 / 0.685 (red); `vpc_tte`'s
//! keep-own-censoring rule gives 0.90 / 0.95 / 0.95, red at 3H/8 and H/2 on this M but
//! within a few datasets of the bound, so `kmmc_replicates_draw_every_censoring_time_from_g`
//! is the test that owns it; a curve that ignores the covariate values gives coverage 1.0
//! and an outside share of 0 (both red). Removing the at-risk floor, everywhere or only
//! from the replicates, changes nothing here (the floor of 10 binds in almost no dataset
//! at these times), so the floor is owned by the unit tests `kmmc_band_floors_each_curve_on_its_own`
//! and `driver_floors_the_band_and_the_observed_curve`.

#[cfg(feature = "survival")]
mod self_consistency {
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::{EventType, ObsRecord, Population, Subject};
    use ferx_core::{vpc_kmmc, vpc_tte, SimulateOptions, VpcGrid, VpcKmmcOptions, VpcTteOptions};
    use rand::{RngExt, SeedableRng};

    /// The data-generating model: `scale = TVLAMBDA`, `loghr = TVBETA * Z`, no IIV.
    const OWN: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  theta TVBETA(0.7, -5.0, 5.0)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA
  loghr  = TVBETA * Z
";
    /// No covariate effect, marginal rate matched: `0.05·E[exp(0.7·Z)] = 0.05·exp(0.245)`.
    const NO_COV: &str = r"
[parameters]
  theta TVLAMBDA(0.06388265, 0.001, 10.0)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA
";
    const LAMBDA: f64 = 0.05;
    const BETA: f64 = 0.7;
    const H: f64 = 30.0;
    const DROP_MAX: f64 = 30.0;
    const N_SUBJ: usize = 100;
    const M: usize = 200;
    const N_SIM: usize = 200;
    const GRID: [f64; 3] = [H / 4.0, 3.0 * H / 8.0, H / 2.0];

    fn draw_dataset(seed: u64) -> Population {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let subjects = (1..=N_SUBJ)
            .map(|i| {
                let (u1, u2): (f64, f64) = (rng.random(), rng.random());
                let z = (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let t_event = -(1.0 - rng.random::<f64>()).ln() / (LAMBDA * (BETA * z).exp());
                let c = (rng.random::<f64>() * DROP_MAX).min(H);
                let (time, event_type) = if t_event <= c {
                    (t_event, EventType::Exact)
                } else {
                    (c, EventType::RightCensored)
                };
                let mut s = Subject {
                    id: i.to_string(),
                    obs_records: vec![ObsRecord::Event {
                        time,
                        event_type,
                        entry_time: 0.0,
                        cmt: 2,
                    }],
                    ..Default::default()
                };
                s.covariates.insert("Z".into(), z);
                s
            })
            .collect();
        Population {
            subjects,
            covariate_names: vec!["Z".into()],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    fn kmmc_opts(m: usize) -> VpcKmmcOptions {
        let mut o = VpcKmmcOptions::new("Z");
        o.vpc.n_sim = N_SIM;
        o.vpc.grid = VpcGrid::Times(GRID.to_vec());
        o.vpc.sim = SimulateOptions {
            seed: Some(m as u64),
            horizon: Some(H),
            ..Default::default()
        };
        o
    }

    /// Per grid time: (datasets inside the band, datasets where obs and band are defined).
    fn tally(src: &str) -> ([usize; 3], [usize; 3]) {
        let model = parse_model_string(src).unwrap();
        let (mut inside, mut defined) = ([0usize; 3], [0usize; 3]);
        for m in 0..M {
            let pop = draw_dataset(2_000_000 + m as u64);
            let v = vpc_kmmc(&model, &pop, &model.default_params, &kmmc_opts(m)).unwrap();
            let s = &v.strata[0];
            for k in 0..3 {
                let (Some(obs), Some(lo), Some(hi)) = (s.obs_on_grid[k], s.lo[k], s.hi[k]) else {
                    continue;
                };
                assert!(obs.is_finite() && lo.is_finite() && hi.is_finite());
                defined[k] += 1;
                if lo <= obs && obs <= hi {
                    inside[k] += 1;
                }
            }
        }
        (inside, defined)
    }

    /// K9. Coverage of the own model at H/4, 3H/8, H/2 inside the 99 % binomial interval
    /// around 0.891, at every grid time, with enough datasets defined at each.
    #[test]
    fn observed_kmmc_coverage_matches_its_expectation() {
        let (inside, defined) = tally(OWN);
        let cov: Vec<f64> = (0..3)
            .map(|k| inside[k] as f64 / defined[k] as f64)
            .collect();
        eprintln!("K9 own model: coverage {cov:?} over {defined:?} datasets");
        for k in 0..3 {
            assert!(
                defined[k] >= 190,
                "only {} datasets defined at {}",
                defined[k],
                GRID[k]
            );
            assert!(
                (LO..=HI).contains(&cov[k]),
                "coverage {} at {} outside [{LO}, {HI}]",
                cov[k],
                GRID[k]
            );
        }
    }

    /// K10. A model without the covariate effect (rate matched) leaves the observed
    /// KMMC outside the band at H/4 in at least half the datasets. The KM-VPC of the
    /// same model on the same data is printed beside it, not asserted: it mostly
    /// covers, which is why the KMMC exists.
    #[test]
    fn kmmc_detects_a_missing_covariate_effect() {
        let (inside, defined) = tally(NO_COV);
        let outside: Vec<f64> = (0..3)
            .map(|k| 1.0 - inside[k] as f64 / defined[k] as f64)
            .collect();
        let model = parse_model_string(NO_COV).unwrap();
        let mut km_inside = [0usize; 3];
        for m in 0..M {
            let pop = draw_dataset(2_000_000 + m as u64);
            let opts = VpcTteOptions {
                n_sim: N_SIM,
                grid: VpcGrid::Times(GRID.to_vec()),
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
                if let (Some(o), Some(lo), Some(hi)) = (s.obs_on_grid[k], s.lo[k], s.hi[k]) {
                    if lo <= o && o <= hi {
                        km_inside[k] += 1;
                    }
                }
            }
        }
        let km_cov: Vec<f64> = km_inside.iter().map(|&n| n as f64 / M as f64).collect();
        eprintln!(
            "K10 no-cov model: KMMC outside-band share {outside:?} over {defined:?}; \
             KM-VPC coverage {km_cov:?}"
        );
        assert!(
            defined[0] >= 190,
            "only {} datasets defined at H/4",
            defined[0]
        );
        assert!(
            outside[0] >= 0.5,
            "the KMMC saw the missing effect in only {} of datasets at H/4",
            outside[0]
        );
    }

    /// The 99 % binomial interval for `M = 200` around 0.891:
    /// `qbinom(c(.005, .995), 200, 0.891) / 200` in R, as PR1's and PR2's files use.
    const LO: f64 = 0.83;
    const HI: f64 = 0.945;
}
