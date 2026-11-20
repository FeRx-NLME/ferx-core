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
//! a fixed time in about 89 % of datasets: a band read from `N` replicates with the type-7
//! quantile covers an exchangeable observation with probability `0.90·(N − 1)/(N + 1)`,
//! which is 0.891 for 200, not the nominal 0.90.
//!
//! Two curves are checked at H/4, H/2 and 3H/4: the mean cumulative function and the
//! order-2 Kaplan–Meier curve on absolute time. The design is chosen so the follow-up
//! step is visible. Measured on macOS arm64 over these 200 datasets (MCF, then order-2
//! KM, each at H/4, H/2, 3H/4):
//!
//! - as shipped: 0.890 / 0.890 / 0.920 and 0.890 / 0.850 / 0.925;
//! - `Observed` acting as `Horizon` (every replicate subject followed to `H`):
//!   0.885 / 0.845 / 0.835 and 0.870 / 0.790 / 0.820, which fails on the order-2 curve
//!   at H/2 and 3H/4. Replicates with no dropout keep more subjects at risk and the
//!   band narrows.
//!
//! Two other seed sets gave a lowest point of 0.840 and 0.855, and pooled over the three
//! sets (600 datasets) every point is between 0.873 and 0.895, around the expected
//! 0.891. This test pins these seeds, and a point falls below the lower bound by chance
//! in 0.33 % of seed sets (`pbinom(165, 200, 0.891)`). So a change that moves the draws can move a point
//! across the bound without being a defect: re-measure the pooled coverage before
//! reading a failure here as one.
//!
//! **A horizon below the follow-up (#2908, plan T5).** The same design with `ω² = 1`,
//! the data still followed to 20 and `sim.horizon = 10`. `vpc_rtte` reads the data on
//! `[0, 10]` only, as the replicates are, so the gap-time bands at 2, 4 and 6 cover
//! the observed curves and the misses fall on both sides. Measured numbers are in the
//! test's doc.

#[cfg(feature = "survival")]
mod self_consistency {
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::{EventType, ObsRecord, Population, Subject};
    use ferx_core::{vpc_rtte, SimulateOptions, VpcGrid, VpcRtteOptions};
    use rand::{RngExt, SeedableRng};

    fn model(omega: f64) -> String {
        format!(
            "
[parameters]
  theta TVLAMBDA(0.15, 0.001, 10.0)
  omega ETA_LAMBDA ~ {omega}

[event_model]
  cmt    = 2
  type   = rtte
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
"
        )
    }
    const LAMBDA: f64 = 0.15;
    const OMEGA: f64 = 0.09;
    const H: f64 = 20.0;
    /// Uniform dropout on `[0, DROP_MAX]`, capped at `H`.
    const DROP_MAX: f64 = 25.0;
    const N_SUBJ: usize = 100;
    const M: usize = 200;
    const N_SIM: usize = 200;

    /// One dataset: per subject a Poisson process of rate `λ·exp(η)`, `η ~ N(0, omega)`,
    /// on `[0, c]` and a closing censoring record at `c = min(dropout, H)`. Returns the
    /// fraction of subjects that left before `H`.
    fn draw_dataset(seed: u64, omega: f64) -> (Population, f64) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut n_early = 0usize;
        let subjects = (1..=N_SUBJ)
            .map(|i| {
                // Box–Muller for η ~ N(0, ω²).
                let (u1, u2): (f64, f64) = (rng.random(), rng.random());
                let z = (-2.0 * (1.0 - u1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let rate = LAMBDA * (omega.sqrt() * z).exp();
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
                Subject { id: i.to_string(), obs_records, ..Default::default() }
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
        let model = parse_model_string(&model(OMEGA)).unwrap();
        let grid = vec![H / 4.0, H / 2.0, 3.0 * H / 4.0];
        // inside[curve][grid point]: curve 0 is the MCF, 1 the order-2 absolute KM.
        let mut inside = [[0usize; 3]; 2];
        let mut early = 0.0;
        for m in 0..M {
            let (pop, e) = draw_dataset(2_000_000 + m as u64, OMEGA);
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
            let curves =
                [(&s.mcf.obs_on_grid, &s.mcf.lo, &s.mcf.hi), (&km2.obs_on_grid, &km2.lo, &km2.hi)];
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
        let cov: Vec<Vec<f64>> =
            inside.iter().map(|c| c.iter().map(|&n| n as f64 / M as f64).collect()).collect();
        eprintln!(
            "left before H: {early:.3}; coverage at H/4, H/2, 3H/4: MCF {:?}, order-2 KM {:?}",
            cov[0], cov[1]
        );
        assert!((0.75..0.85).contains(&early), "dropout design drifted: {early}");
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

    /// T5 (#2908). `ω² = 1`, data followed to `H = 20`, `sim.horizon = 10`, grid 2, 4, 6,
    /// orders to 3. Each gap-time curve (orders 2 and 3) covers at each point within
    /// `[LO, HI]`, and of its misses pooled over the three points at least a quarter fall
    /// below the band.
    ///
    /// The second condition is what catches the defect: with the observed streams left
    /// uncut (as shipped before #2908) the observed gap-time curves read follow-up the
    /// replicates do not have and the misses fall almost all above the band (plan §0 M7:
    /// a below-share of 0.08–0.19), while the coverage stays inside `[LO, HI]` (0.86–0.915),
    /// so a coverage-only test cannot fail on it. Under symmetry a below-share under 0.25
    /// with about 60 misses has probability about 3e-5 (`pbinom(15, 62, 0.5)`).
    ///
    /// Measured with these seeds (datasets `3_000_000 + m`, replicates `3_000_000 ^ m`),
    /// macOS arm64:
    ///
    /// | curve | coverage at 2 / 4 / 6 | misses below / above | below-share |
    /// |---|---|---|---|
    /// | gap 2 | 0.865 / 0.865 / 0.894 | 40 / 35 | 0.53 |
    /// | gap 3 | 0.905 / 0.905 / 0.896 | 27 / 30 | 0.47 |
    /// | gap 2, observed streams not cut | 0.895 / 0.880 / 0.870 | 11 / 60 | **0.15** |
    /// | gap 3, observed streams not cut | 0.890 / 0.890 / 0.940 | 1 / 55 | **0.02** |
    ///
    /// Linux x86_64 (CI slow-tests run 43616779609, at `b28f22101`) printed the cut rows
    /// and the MCF / absolute-2 control with every digit, undefined count and miss count
    /// identical to the macOS arm64 numbers above. Since #2930 (a curve that reached 0
    /// is undefined past its last time) the gap 3 row reads 0.896 at 6 with 27 / 30
    /// misses, where it read 0.911 with 25 / 30; re-measured on Linux aarch64
    /// (`tools/linux-test.sh`) and macOS arm64 with identical digits. The other rows
    /// did not move; the two "not cut" rows predate #2930 and were not re-run.
    ///
    /// Coverage is over the datasets where the observed value and the band are both
    /// defined: at most 17 of 200 are not (gap 3 at 6, where a dataset's observed gap
    /// curve, or too many replicate gap curves, end first; 10 before #2930); the bound
    /// `MAX_UNDEFINED` is twice that.
    ///
    /// The MCF and absolute order 2, which the cut does not move on `[0, H]` (plan §0 M3),
    /// are printed as a control: 0.860 / 0.875 / 0.860 and 0.840 / 0.875 / 0.890, the
    /// same digits with the cut and without it.
    #[test]
    fn gap_bands_cover_when_the_horizon_is_below_the_follow_up() {
        const HZ: f64 = 10.0;
        const BASE: u64 = 3_000_000;
        const MAX_UNDEFINED: usize = 34;
        let model = parse_model_string(&model(1.0)).unwrap();
        let grid = vec![HZ / 5.0, 2.0 * HZ / 5.0, 3.0 * HZ / 5.0];
        let names = ["MCF", "absolute 2", "gap 2", "gap 3"];
        // [curve][grid point]: (inside, below, above, undefined). Undefined: the observed
        // curve ended before the grid time (also at 0, #2930), or too few replicate
        // curves reach it for a band.
        let mut tally = [[(0usize, 0usize, 0usize, 0usize); 3]; 4];
        for m in 0..M {
            let (pop, _) = draw_dataset(BASE + m as u64, 1.0);
            let opts = VpcRtteOptions {
                n_sim: N_SIM,
                grid: VpcGrid::Times(grid.clone()),
                max_order: Some(3),
                sim: SimulateOptions {
                    seed: Some(BASE ^ m as u64),
                    horizon: Some(HZ),
                    ..Default::default()
                },
                ..Default::default()
            };
            let v = vpc_rtte(&model, &pop, &model.default_params, &opts).unwrap();
            assert_eq!(v.horizon, HZ);
            let s = &v.strata[0];
            let (a2, g2, g3) = (&s.absolute[1], &s.gap[0], &s.gap[1]);
            assert_eq!((a2.order, g2.order, g3.order), (2, 2, 3));
            let curves = [
                (&s.mcf.obs_on_grid, &s.mcf.lo, &s.mcf.hi),
                (&a2.obs_on_grid, &a2.lo, &a2.hi),
                (&g2.obs_on_grid, &g2.lo, &g2.hi),
                (&g3.obs_on_grid, &g3.lo, &g3.hi),
            ];
            for (j, (obs, lo, hi)) in curves.into_iter().enumerate() {
                for k in 0..3 {
                    let t = &mut tally[j][k];
                    let (Some(obs), Some(lo), Some(hi)) = (obs[k], lo[k], hi[k]) else {
                        t.3 += 1;
                        continue;
                    };
                    assert!(obs.is_finite() && lo.is_finite() && hi.is_finite());
                    if obs < lo {
                        t.1 += 1;
                    } else if obs > hi {
                        t.2 += 1;
                    } else {
                        t.0 += 1;
                    }
                }
            }
        }
        for (name, row) in names.iter().zip(&tally) {
            let cov: Vec<f64> = row.iter().map(|t| t.0 as f64 / (M - t.3) as f64).collect();
            let undefined: Vec<usize> = row.iter().map(|t| t.3).collect();
            let (below, above) = row.iter().fold((0, 0), |(b, a), t| (b + t.1, a + t.2));
            eprintln!(
                "{name}: coverage at 2, 4, 6 {cov:?}; undefined {undefined:?}; misses below \
                 {below}, above {above}"
            );
        }
        for (name, row) in names.iter().zip(&tally).skip(2) {
            for (k, t) in row.iter().enumerate() {
                assert!(
                    t.3 <= MAX_UNDEFINED,
                    "{name}: {} of {M} datasets undefined at grid time {}",
                    t.3,
                    grid[k]
                );
                let c = t.0 as f64 / (M - t.3) as f64;
                assert!(
                    (LO..=HI).contains(&c),
                    "{name}: coverage {c} at grid time {} outside [{LO}, {HI}]",
                    grid[k]
                );
            }
            let (below, above) = row.iter().fold((0, 0), |(b, a), t| (b + t.1, a + t.2));
            let share = below as f64 / (below + above) as f64;
            assert!(
                below + above >= 30 && share >= 0.25,
                "{name}: {below} misses below and {above} above the band, below-share {share:.2} \
                 under 0.25: the observed curve reads follow-up past the horizon"
            );
        }
    }

    /// T5′ (#2930). A Kaplan–Meier curve is undefined past its last time, also when it
    /// reached 0. The T13 design at `ω² = 1` (datasets `2_000_000 + m`, replicates
    /// `2_000_000 ^ m`, `sim.horizon = H = 20`), orders to 3, grid 10, 12, 15.
    ///
    /// **A1**, deterministic: for every order curve (absolute 1–3, gap 2–3) and every grid
    /// time past the observed curve's last time, the observed value is `None`. The
    /// straddle is asserted too: at least 50 of those cells have an observed curve that
    /// reached 0, the cells the rule before #2930 read as `Some(0.0)`. Measured: 272 cells
    /// past the last time, 104 of them at zero.
    ///
    /// **A2**, the one-sided-miss signal: for gap order 3, at least a quarter of the misses
    /// pooled over the three times fall below the band. Before #2930 a replicate that
    /// reached 0 added a 0 to the band past its last time, so once enough did the lower
    /// edge was 0, the observed curve could not fall below it and the misses fell above.
    /// Measured with these seeds: this rule, 30 below / 19 above (share 0.61) on Linux
    /// aarch64 (`tools/linux-test.sh`) and macOS arm64 alike; before #2930, 4 / 23 (0.15);
    /// carrying the last value forward, 4 / 31 (0.11); the band keeping the zeros while
    /// the observed curve does not, 2 / 23 (0.08); all three macOS arm64. Under symmetry a
    /// below-share under 0.25 with 49 misses has probability about 2e-4.
    ///
    /// A separate sample-size floor of 20 misses comes first, so the share is read only
    /// where it can mean something (with 20 misses a share under 0.25 has probability
    /// about 6e-3 under symmetry). It sits below every measured count, 25 to 49, so it
    /// passes under each of the rules above and the share assertion alone is what fails
    /// on them; a floor failure says the design lost its misses, not that they are
    /// one-sided.
    #[test]
    fn gap_band_past_a_curve_last_time_has_misses_on_both_sides() {
        const BASE: u64 = 2_000_000;
        let model = parse_model_string(&model(1.0)).unwrap();
        let grid = vec![10.0, 12.0, 15.0];
        let (mut past_end, mut past_end_at_zero) = (0usize, 0usize);
        let (mut below, mut above) = (0usize, 0usize);
        for m in 0..M {
            let (pop, _) = draw_dataset(BASE + m as u64, 1.0);
            let opts = VpcRtteOptions {
                n_sim: N_SIM,
                grid: VpcGrid::Times(grid.clone()),
                max_order: Some(3),
                sim: SimulateOptions {
                    seed: Some(BASE ^ m as u64),
                    horizon: Some(H),
                    ..Default::default()
                },
                ..Default::default()
            };
            let v = vpc_rtte(&model, &pop, &model.default_params, &opts).unwrap();
            let s = &v.strata[0];
            assert_eq!((s.absolute.len(), s.gap.len()), (3, 2));
            // A1 over every order curve.
            for b in s.absolute.iter().chain(&s.gap) {
                let Some(&last) = b.observed.time.last() else { continue };
                for (k, &t) in grid.iter().enumerate() {
                    if t <= last {
                        continue;
                    }
                    past_end += 1;
                    if b.observed.surv.last() == Some(&0.0) {
                        past_end_at_zero += 1;
                    }
                    assert_eq!(
                        b.obs_on_grid[k], None,
                        "dataset {m}, order {}: observed defined at {t}, past its last time {last}",
                        b.order
                    );
                }
            }
            // A2 on gap order 3.
            let g3 = &s.gap[1];
            assert_eq!(g3.order, 3);
            for k in 0..grid.len() {
                let (Some(obs), Some(lo), Some(hi)) = (g3.obs_on_grid[k], g3.lo[k], g3.hi[k])
                else {
                    continue;
                };
                assert!(obs.is_finite() && lo.is_finite() && hi.is_finite());
                if obs < lo {
                    below += 1;
                } else if obs > hi {
                    above += 1;
                }
            }
        }
        let share = below as f64 / (below + above) as f64;
        eprintln!(
            "past the observed last time: {past_end} cells, {past_end_at_zero} at zero; gap 3 \
             misses at 10, 12, 15: below {below}, above {above}, below-share {share:.2}"
        );
        assert!(
            past_end_at_zero >= 50,
            "only {past_end_at_zero} reached-0 cells past the last time: A1 no longer straddles"
        );
        assert!(
            below + above >= 20,
            "gap 3: only {} misses ({below} below, {above} above) at 10, 12, 15, under the \
             floor of 20: too few to read a below-share, the design has drifted",
            below + above
        );
        assert!(
            share >= 0.25,
            "gap 3: {below} misses below and {above} above the band, below-share {share:.2} \
             under 0.25: replicates that reached 0 hold the band's lower edge at 0"
        );
    }

    /// The 99 % binomial interval for `M = 200` datasets around the exchangeable
    /// expectation `0.90·(N_SIM − 1)/(N_SIM + 1) = 0.891` (see the file doc):
    /// `qbinom(c(.005, .995), 200, 0.891) / 200` in R (review r1 row 2).
    const LO: f64 = 0.83;
    const HI: f64 = 0.945;
}
