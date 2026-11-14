//! Tier 2: the Kaplan–Meier mean covariate against R's `survival::survfit` risk sets
//! (#1886 PR3).
//!
//! NONMEM has no Kaplan–Meier estimator, so there is no NONMEM spelling of this
//! quantity. `tests/fixtures/kmmc_survfit/make_reference.R` (survival 3.8.12, R 4.6.1;
//! regenerates both CSVs byte for byte) draws 120 subjects in two strata with
//! whole-unit times, a weight in hundredths and a 0/1 sex. For each survfit row it
//! writes `n.risk`, `n.event`, `n.censor` and the mean of `wt` and of `sex` over the
//! subjects with time `≥ t`, after checking that this set has exactly `n.risk` members.
//! 15 of the 32 rows carry tied events and 12 an event and a censoring at one time,
//! which is where the risk-set convention decides the mean. R's double mean is within
//! 1.25e-16 relative of the exact fraction (plan M4), so the reference outranks the
//! bound below.
//!
//! Both the kernel (`kaplan_meier_mean_covariate`, per stratum) and the entry point
//! (`vpc_kmmc`'s `observed` curves, with its own stratification) are compared, so a
//! defect in the strata or in the driver's covariate lookup is in reach, not only one
//! in the counting pass. (vpc 1.2.4's `compute_kmmc` computes every stratum from all
//! the data; mixing the strata like that changes stratum B's means here.)

#[cfg(feature = "survival")]
mod kmmc_reference {
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::stats::km::{kaplan_meier_mean_covariate, KmmcCurve};
    use ferx_core::types::{EventType, ObsRecord, Population, Subject};
    use ferx_core::{vpc_kmmc, SimulateOptions, VpcKmmcOptions, VpcStratify};
    use std::collections::BTreeMap;

    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/kmmc_survfit");

    /// The measured worst error against R is printed by the test. A mean over a set is
    /// a finite sum and one division, so the engines differ only in summation order
    /// (ferx sums a suffix in `f64`; R's `mean` accumulates in long double and corrects
    /// in a second pass). Measured on macOS arm64 at the commit that added this test:
    /// worst |Δmean| = 2.132e-14, on a weight mean near 71, where one ULP is 1.42e-14.
    /// The bound is 4.7× that. The smallest mutation caught here (censor-first, or a
    /// stratum's subjects read into the other) moves a mean by more than 0.01.
    const TOL: f64 = 1e-13;

    const MODEL: &str = r"
[parameters]
  theta TVLAMBDA(0.15, 0.001, 10.0)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA
";

    struct Row {
        time: f64,
        n_risk: usize,
        n_event: usize,
        n_censor: usize,
        mean_wt: f64,
        mean_sex: f64,
    }

    struct Data {
        stratum: String,
        time: f64,
        event: bool,
        wt: f64,
        sex: f64,
    }

    fn read_csv(name: &str) -> Vec<Vec<String>> {
        let text = std::fs::read_to_string(format!("{DIR}/{name}")).unwrap();
        text.lines()
            .skip(1)
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
            .collect()
    }

    fn data() -> Vec<(String, Data)> {
        read_csv("data.csv")
            .into_iter()
            .map(|r| {
                (
                    r[0].clone(),
                    Data {
                        stratum: r[1].clone(),
                        time: r[2].parse().unwrap(),
                        event: r[3] == "1",
                        wt: r[4].parse().unwrap(),
                        sex: r[5].parse().unwrap(),
                    },
                )
            })
            .collect()
    }

    fn reference() -> BTreeMap<String, Vec<Row>> {
        let mut by: BTreeMap<String, Vec<Row>> = BTreeMap::new();
        for r in read_csv("reference.csv") {
            by.entry(r[0].clone()).or_default().push(Row {
                time: r[1].parse().unwrap(),
                n_risk: r[2].parse().unwrap(),
                n_event: r[3].parse().unwrap(),
                n_censor: r[4].parse().unwrap(),
                mean_wt: r[5].parse().unwrap(),
                mean_sex: r[6].parse().unwrap(),
            });
        }
        by
    }

    /// The fixture as a population: the stratum as `STRAT` (A = 0, B = 1), `WT`, `SEX`.
    fn population() -> Population {
        let subjects = data()
            .into_iter()
            .map(|(id, d)| {
                let mut s = Subject {
                    id,
                    obs_records: vec![ObsRecord::Event {
                        time: d.time,
                        event_type: if d.event {
                            EventType::Exact
                        } else {
                            EventType::RightCensored
                        },
                        entry_time: 0.0,
                        cmt: 2,
                    }],
                    ..Default::default()
                };
                let strat = match d.stratum.as_str() {
                    "A" => 0.0,
                    "B" => 1.0,
                    other => panic!("unknown stratum {other}"),
                };
                s.covariates.insert("STRAT".into(), strat);
                s.covariates.insert("WT".into(), d.wt);
                s.covariates.insert("SEX".into(), d.sex);
                s
            })
            .collect();
        Population {
            subjects,
            covariate_names: vec!["STRAT".into(), "WT".into(), "SEX".into()],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    /// Compare one curve with the reference rows; returns the worst |Δmean|.
    fn compare(at: &str, got: &KmmcCurve, want: &[Row], pick: fn(&Row) -> f64) -> f64 {
        assert_eq!(got.time.len(), want.len(), "{at}: row count");
        let mut worst = 0.0_f64;
        for (j, w) in want.iter().enumerate() {
            let at = format!("{at}, t = {}", w.time);
            assert_eq!(got.time[j], w.time, "{at}: time");
            assert_eq!(got.n_risk[j], w.n_risk, "{at}: n.risk");
            assert_eq!(got.n_event[j], w.n_event, "{at}: n.event");
            assert_eq!(got.n_censor[j], w.n_censor, "{at}: n.censor");
            let d = (got.mean[j] - pick(w)).abs();
            assert!(d.is_finite(), "{at}: non-finite mean {}", got.mean[j]);
            assert!(d <= TOL, "{at}: mean {} vs R {}", got.mean[j], pick(w));
            worst = worst.max(d);
        }
        worst
    }

    const COVS: [(&str, fn(&Row) -> f64); 2] =
        [("WT", |r: &Row| r.mean_wt), ("SEX", |r: &Row| r.mean_sex)];

    /// K5, kernel half. Every row of both strata, for both covariates.
    #[test]
    fn kmmc_kernel_matches_survfit_risk_set_means() {
        let reference = reference();
        assert_eq!(reference.keys().collect::<Vec<_>>(), ["A", "B"]);
        let rows = data();
        let mut worst = 0.0_f64;
        for (label, want) in &reference {
            let d: Vec<&Data> = rows
                .iter()
                .map(|(_, d)| d)
                .filter(|d| &d.stratum == label)
                .collect();
            let t: Vec<f64> = d.iter().map(|d| d.time).collect();
            let e: Vec<bool> = d.iter().map(|d| d.event).collect();
            for (name, pick) in COVS {
                let v: Vec<f64> = d
                    .iter()
                    .map(|d| if name == "WT" { d.wt } else { d.sex })
                    .collect();
                let c = kaplan_meier_mean_covariate(&t, &e, &v).unwrap();
                let w = compare(&format!("kernel {label} {name}"), &c, want, pick);
                assert!(w.is_finite());
                worst = worst.max(w);
            }
        }
        let tied = reference
            .values()
            .flatten()
            .filter(|r| r.n_event > 1)
            .count();
        let mixed = reference
            .values()
            .flatten()
            .filter(|r| r.n_event > 0 && r.n_censor > 0)
            .count();
        assert!(
            tied >= 10 && mixed >= 5,
            "fixture lost its ties: {tied} tied, {mixed} mixed"
        );
        eprintln!("kmmc kernel vs R: worst |Δmean| = {worst:.3e}");
    }

    /// K5, driver half. `vpc_kmmc` stratified by `STRAT`, for both covariates: its
    /// `observed` curves equal the reference per stratum, so stratum labels, the
    /// covariate lookup by subject and the per-stratum split are all checked.
    #[test]
    fn vpc_kmmc_observed_curves_match_survfit_risk_set_means() {
        let reference = reference();
        let model = parse_model_string(MODEL).unwrap();
        let p = population();
        let mut worst = 0.0_f64;
        for (name, pick) in COVS {
            let mut o = VpcKmmcOptions::new(name);
            o.vpc.n_sim = 20;
            o.vpc.stratify = VpcStratify::Covariate("STRAT".into());
            o.vpc.sim = SimulateOptions {
                seed: Some(1),
                ..Default::default()
            };
            let r = vpc_kmmc(&model, &p, &model.default_params, &o).unwrap();
            let labels: Vec<&str> = r.strata.iter().map(|s| s.label.as_str()).collect();
            assert_eq!(labels, ["STRAT=0", "STRAT=1"]);
            for (s, key) in r.strata.iter().zip(["A", "B"]) {
                assert_eq!(s.n_subjects, 60);
                let w = compare(
                    &format!("vpc_kmmc {} {name}", s.label),
                    &s.observed,
                    &reference[key],
                    pick,
                );
                assert!(w.is_finite());
                worst = worst.max(w);
            }
        }
        eprintln!("vpc_kmmc vs R: worst |Δmean| = {worst:.3e}");
    }
}
