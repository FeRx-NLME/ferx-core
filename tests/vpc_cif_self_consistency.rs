//! Tier 2: the cumulative-incidence VPC band covers data drawn from its own model, and
//! misses data whose causes it has swapped (#1887, plan T9).
//!
//! The oracle is self-consistency, as for `vpc_tte_self_consistency`: NONMEM has no
//! nonparametric cumulative incidence and no VPC. Each of `M` "observed" datasets is
//! drawn **here**, with this file's own RNG and closed form, outside the simulator: two
//! exponential causes, λ_A = 0.06 (cmt 2) and λ_B = 0.04 (cmt 3), scaled by one shared
//! log-normal frailty (ω² = 0.09), uniform dropout on `[0, 40]` and an administrative end
//! at H = 30. `vpc_cif` then simulates `N` replicates of each design from the same model.
//! If the bands and the censoring are right, the observed `F_A`, `F_B` and `S` fall inside
//! their 5–95 % bands at a fixed time in about 89 % of datasets: a type-7 band from `N`
//! replicates covers an exchangeable observation with probability
//! `0.90·(N − 1)/(N + 1) = 0.891` for 200.
//!
//! **The straddle.** The same datasets are checked against replicates drawn with the two
//! rates **swapped**. The all-cause hazard is unchanged, so the survival band still covers
//! (≥ 0.83), but the cause bands now sit on the wrong cause and cover in under half the
//! datasets. A check that cannot tell the causes apart passes the first half and fails the
//! second; one that reads the wrong curve fails the first.
//!
//! Measured on macOS arm64 (coverage at H/4, H/2, 3H/4; the test prints them): `F_A`
//! 0.88 / 0.85 / 0.874, `F_B` 0.905 / 0.885 / 0.869, `S` 0.89 / 0.87 / 0.914, all 9 inside
//! the interval. Swapped: `F_A` 0.27 / 0.12 / 0.041, `F_B` 0.25 / 0.11 / 0.061, `S`
//! 0.93 / 0.89 / 0.898. Two of the 200 datasets (two of the 100) had nobody under
//! follow-up at 3H/4 and were left out of that cell. Mutation-checked: replacing the
//! replicates' estimator with `1 − KM_k` (the observed curve unchanged) drops `F_A` to
//! 0.75 / 0.405 / 0.116 and fails.

#[cfg(feature = "survival")]
mod self_consistency {
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::{EventType, ObsRecord, Population, Subject};
    use ferx_core::{vpc_cif, CifVpc, SimulateOptions, VpcCifOptions, VpcGrid};
    use rand::{RngExt, SeedableRng};

    const MODEL: &str = r"
[parameters]
  theta TVLAMBDA_A(0.06, 0.001, 10.0)
  theta TVLAMBDA_B(0.04, 0.001, 10.0)
  omega ETA ~ 0.09

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A * exp(ETA)

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B * exp(ETA)
";
    const LAMBDA_A: f64 = 0.06;
    const LAMBDA_B: f64 = 0.04;
    const OMEGA: f64 = 0.09;
    const H: f64 = 30.0;
    /// Uniform dropout on `[0, DROP_MAX]`.
    const DROP_MAX: f64 = 40.0;
    const N_SUBJ: usize = 100;
    const M: usize = 200;
    /// The swapped-rates half reads the first `M_SWAP` datasets; its effect is large.
    const M_SWAP: usize = 100;
    const N_SIM: usize = 200;

    /// One dataset: per subject a frailty `exp(η)`, both latent cause times, and a
    /// censoring at `min(dropout, H)`. Every subject gets a record on both causes at its
    /// all-cause time, the event on the earlier cause when it precedes the censoring.
    fn draw_dataset(seed: u64) -> Population {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let subjects = (1..=N_SUBJ)
            .map(|i| {
                // Box–Muller for η ~ N(0, ω²).
                let (u1, u2): (f64, f64) = (rng.random(), rng.random());
                let z = (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let frailty = (OMEGA.sqrt() * z).exp();
                let a = -(1.0 - rng.random::<f64>()).ln() / (LAMBDA_A * frailty);
                let b = -(1.0 - rng.random::<f64>()).ln() / (LAMBDA_B * frailty);
                let c = (rng.random::<f64>() * DROP_MAX).min(H);
                let t = a.min(b);
                let (time, winner) = if t <= c { (t, if a < b { 2 } else { 3 }) } else { (c, 0) };
                let rec = |cmt: usize| ObsRecord::Event {
                    time,
                    event_type: if winner == cmt {
                        EventType::Exact
                    } else {
                        EventType::RightCensored
                    },
                    entry_time: 0.0,
                    cmt,
                };
                Subject {
                    id: i.to_string(),
                    obs_records: vec![rec(2), rec(3)],
                    ..Default::default()
                }
            })
            .collect();
        Population {
            subjects,
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    /// Coverage tallies per curve (`F_A`, `F_B`, `S`) and grid time: datasets inside the
    /// band, and datasets where the cell is defined at all. With a total hazard of 0.1 a
    /// dataset now and then has nobody under follow-up at 3H/4 (about 1 % per dataset),
    /// and past its last time the observed curve is undefined (#2930), so that cell is
    /// left out rather than counted as a miss. `skipped` is capped in the test.
    #[derive(Default)]
    struct Tally {
        hit: [[usize; 3]; 3],
        defined: [[usize; 3]; 3],
    }

    impl Tally {
        fn add(&mut self, v: &CifVpc) {
            let s = &v.strata[0];
            let bands = [&s.causes[0].band, &s.causes[1].band, &s.survival];
            for (c, b) in bands.iter().enumerate() {
                for g in 0..3 {
                    let (Some(obs), Some(lo), Some(hi)) = (b.obs_on_grid[g], b.lo[g], b.hi[g])
                    else {
                        continue;
                    };
                    assert!(obs.is_finite() && lo.is_finite() && hi.is_finite());
                    self.defined[c][g] += 1;
                    self.hit[c][g] += usize::from(lo <= obs && obs <= hi);
                }
            }
        }

        fn coverage(&self) -> [[f64; 3]; 3] {
            let mut out = [[0.0; 3]; 3];
            for c in 0..3 {
                for g in 0..3 {
                    out[c][g] = self.hit[c][g] as f64 / self.defined[c][g] as f64;
                }
            }
            out
        }

        fn skipped(&self, n: usize) -> usize {
            self.defined.iter().flatten().map(|&d| n - d).max().unwrap_or(0)
        }
    }

    #[test]
    fn cif_coverage_matches_its_expectation_and_misses_swapped_causes() {
        let model = parse_model_string(MODEL).unwrap();
        let mut swapped = model.default_params.clone();
        swapped.theta.swap(0, 1);
        let grid = vec![H / 4.0, H / 2.0, 3.0 * H / 4.0];
        let (mut own, mut swap_t) = (Tally::default(), Tally::default());
        for m in 0..M {
            let pop = draw_dataset(1_887_000 + m as u64);
            let opts = VpcCifOptions {
                n_sim: N_SIM,
                grid: VpcGrid::Times(grid.clone()),
                sim: SimulateOptions {
                    seed: Some(m as u64),
                    horizon: Some(H),
                    ..Default::default()
                },
                ..Default::default()
            };
            let v = vpc_cif(&model, &pop, &model.default_params, &opts).unwrap();
            assert_eq!(v.cmts, [2, 3]);
            own.add(&v);
            if m < M_SWAP {
                swap_t.add(&vpc_cif(&model, &pop, &swapped, &opts).unwrap());
            }
        }
        let (cov, swap) = (own.coverage(), swap_t.coverage());
        eprintln!("coverage F_A, F_B, S at H/4, H/2, 3H/4: {cov:?}");
        eprintln!("swapped causes, same: {swap:?}");
        eprintln!(
            "cells left out: {} of {M}, {} of {M_SWAP}",
            own.skipped(M),
            swap_t.skipped(M_SWAP)
        );
        assert!(own.skipped(M) <= 6 && swap_t.skipped(M_SWAP) <= 4, "too many undefined cells");
        let names = ["F_A", "F_B", "S"];
        for c in 0..3 {
            for g in 0..3 {
                let (k, t) = (cov[c][g], grid[g]);
                assert!(
                    (LO..=HI).contains(&k),
                    "{}: coverage {k} at {t} outside [{LO}, {HI}]",
                    names[c]
                );
            }
        }
        for g in 0..3 {
            let t = grid[g];
            assert!(
                swap[0][g] < 0.5 && swap[1][g] < 0.5,
                "swapped causes covered at {t}: {swap:?}"
            );
            assert!(swap[2][g] >= LO, "swapped causes broke the survival band at {t}: {swap:?}");
        }
    }

    /// The 99 % binomial interval for `M = 200` datasets around the exchangeable
    /// expectation `0.90·(N_SIM − 1)/(N_SIM + 1) = 0.891`:
    /// `qbinom(c(.005, .995), 200, 0.891) / 200` in R.
    const LO: f64 = 0.83;
    const HI: f64 = 0.945;
}
