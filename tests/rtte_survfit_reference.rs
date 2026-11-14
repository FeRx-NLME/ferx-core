//! Tier 2: the observed curves of `vpc_rtte` against R's `survival::survfit` (#1886 PR2).
//!
//! NONMEM has no Kaplan–Meier or mean-cumulative-function estimator, so survfit is the
//! external reference. `tests/fixtures/rtte_survfit/make_reference.R` draws 80 subjects
//! in two strata with whole-unit times: 13 have no event, 19 have an event at their own
//! follow-up end, and 32 of the mean-cumulative-function rows carry more than one event.
//! No subject has two events at one time, because survfit refuses `stop == start`; that
//! case is pinned by the exact closed form in `src/stats/km_tests.rs`.
//!
//! The entry point itself is run on the data, with its own stratification, and its
//! `observed` curves are compared: the mean cumulative function against
//! `survfit(Surv(start, stop, status) ~ 1, id = id)$cumhaz`, and the Kaplan–Meier curves
//! by event order (absolute time 1..3, gap time 2..3) against `survfit(Surv(t, d) ~ 1)`
//! on per-order times built in R. So a defect in the strata, in the follow-up ends or
//! in the per-order risk sets is in reach, not only one in the counting pass.

#[cfg(feature = "survival")]
mod survfit_reference {
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::{EventType, ObsRecord, Population, Subject};
    use ferx_core::{vpc_rtte, RtteVpc, SimulateOptions, VpcRtteOptions, VpcStratify};
    use std::collections::BTreeMap;

    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/rtte_survfit");

    /// The measured worst error against survfit is printed by the test. The mean
    /// cumulative function is a finite sum and Kaplan–Meier a finite product over the
    /// same rows, so the engines differ only in rounding. Measured on macOS arm64 at the
    /// commit that added this test: worst |ΔMCF| = 0, worst |ΔS| = 1.110e-16 (1 ULP below
    /// 1). The bound is 9× the larger. Mutation-checked: letting an event leave the
    /// mean cumulative function's risk set, each wrong gap-time risk set or origin,
    /// pooling the strata, and moving one follow-up end in `data.csv` each fail this test.
    const TOL: f64 = 1e-15;

    const MODEL: &str = r"
[parameters]
  theta TVLAMBDA(0.15, 0.001, 10.0)

[event_model]
  cmt    = 2
  type   = rtte
  family = exponential
  scale  = TVLAMBDA
";

    struct Row {
        time: f64,
        n_risk: usize,
        n_event: usize,
        n_censor: usize,
        value: f64,
    }

    fn read_csv(name: &str) -> Vec<Vec<String>> {
        let text = std::fs::read_to_string(format!("{DIR}/{name}")).unwrap();
        text.lines()
            .skip(1)
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
            .collect()
    }

    /// The fixture as a population: one subject per id in file order, its records in
    /// file order, and the stratum as the covariate `STRAT` (A = 0, B = 1).
    fn population() -> Population {
        let mut subjects: Vec<Subject> = Vec::new();
        for r in read_csv("data.csv") {
            if subjects.last().map(|s| s.id.as_str()) != Some(r[0].as_str()) {
                let mut s = Subject {
                    id: r[0].clone(),
                    ..Default::default()
                };
                let strat = match r[1].as_str() {
                    "A" => 0.0,
                    "B" => 1.0,
                    other => panic!("unknown stratum {other}"),
                };
                s.covariates.insert("STRAT".into(), strat);
                subjects.push(s);
            }
            let event_type = if r[3] == "1" {
                EventType::Exact
            } else {
                EventType::RightCensored
            };
            subjects
                .last_mut()
                .unwrap()
                .obs_records
                .push(ObsRecord::Event {
                    time: r[2].parse().unwrap(),
                    event_type,
                    entry_time: 0.0,
                    cmt: 2,
                });
        }
        Population {
            subjects,
            covariate_names: vec!["STRAT".into()],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        }
    }

    /// The survfit rows by `(stratum, curve, order)`.
    fn reference() -> BTreeMap<(String, String, usize), Vec<Row>> {
        let mut by: BTreeMap<(String, String, usize), Vec<Row>> = BTreeMap::new();
        for r in read_csv("reference.csv") {
            by.entry((r[0].clone(), r[1].clone(), r[2].parse().unwrap()))
                .or_default()
                .push(Row {
                    time: r[3].parse().unwrap(),
                    n_risk: r[4].parse().unwrap(),
                    n_event: r[5].parse().unwrap(),
                    n_censor: r[6].parse().unwrap(),
                    value: r[7].parse().unwrap(),
                });
        }
        by
    }

    /// Compare one curve's rows with survfit's; `n_censor` is `None` for the mean
    /// cumulative function, whose curve has no such column. Returns the worst |Δvalue|.
    fn compare(
        at: &str,
        want: &[Row],
        time: &[f64],
        n_risk: &[usize],
        n_event: &[usize],
        n_censor: Option<&[usize]>,
        value: &[f64],
    ) -> f64 {
        assert_eq!(time.len(), want.len(), "{at}: row count");
        let mut worst = 0.0_f64;
        for (j, w) in want.iter().enumerate() {
            let at = format!("{at}, t = {}", w.time);
            assert_eq!(time[j], w.time, "{at}: time");
            assert_eq!(n_risk[j], w.n_risk, "{at}: n.risk");
            assert_eq!(n_event[j], w.n_event, "{at}: n.event");
            if let Some(c) = n_censor {
                assert_eq!(c[j], w.n_censor, "{at}: n.censor");
            }
            let d = (value[j] - w.value).abs();
            assert!(d.is_finite(), "{at}: non-finite");
            assert!(d <= TOL, "{at}: {} vs survfit {}", value[j], w.value);
            worst = worst.max(d);
        }
        worst
    }

    fn run() -> RtteVpc {
        let model = parse_model_string(MODEL).unwrap();
        let opts = VpcRtteOptions {
            n_sim: 20,
            max_order: Some(3),
            stratify: VpcStratify::Covariate("STRAT".into()),
            sim: SimulateOptions {
                seed: Some(1),
                ..Default::default()
            },
            ..Default::default()
        };
        vpc_rtte(&model, &population(), &model.default_params, &opts).unwrap()
    }

    /// T12. Every row of every observed curve in both strata equals survfit's: the
    /// integer columns exactly, the estimate within the measured `TOL`.
    #[test]
    fn vpc_rtte_observed_curves_match_survfit_per_stratum() {
        let v = run();
        let reference = reference();
        assert_eq!(reference.len(), 2 * 6, "two strata of six curves");
        let labels: Vec<&str> = v.strata.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, ["STRAT=0", "STRAT=1"]);
        let (mut worst_mcf, mut worst_km) = (0.0_f64, 0.0_f64);
        for (s, name) in v.strata.iter().zip(["A", "B"]) {
            let key = |curve: &str, order: usize| (name.to_string(), curve.to_string(), order);
            assert_eq!(s.n_subjects, 40, "stratum {name}");
            let m = &s.mcf.observed;
            let d = compare(
                &format!("stratum {name}, mcf"),
                &reference[&key("mcf", 0)],
                &m.time,
                &m.n_risk,
                &m.n_event,
                None,
                &m.mcf,
            );
            assert!(d.is_finite());
            worst_mcf = worst_mcf.max(d);
            assert_eq!((s.absolute.len(), s.gap.len()), (3, 2));
            let curves = s
                .absolute
                .iter()
                .map(|b| ("abs", b))
                .chain(s.gap.iter().map(|b| ("gap", b)));
            for (curve, b) in curves {
                let c = &b.observed;
                let d = compare(
                    &format!("stratum {name}, {curve} order {}", b.order),
                    &reference[&key(curve, b.order)],
                    &c.time,
                    &c.n_risk,
                    &c.n_event,
                    Some(&c.n_censor),
                    &c.surv,
                );
                assert!(d.is_finite());
                worst_km = worst_km.max(d);
            }
        }
        // The fixture must exercise what decides the answer: tied events across
        // subjects, and an event at a follow-up end, which shows as a next-order gap
        // censored at 0.
        let tied = reference
            .iter()
            .filter(|(k, _)| k.1 == "mcf")
            .flat_map(|(_, rows)| rows)
            .filter(|r| r.n_event > 1)
            .count();
        let censored_at_zero = reference
            .iter()
            .filter(|(k, _)| k.1 == "gap")
            .flat_map(|(_, rows)| rows)
            .filter(|r| r.time == 0.0 && r.n_censor > 0)
            .count();
        assert!(
            tied >= 10 && censored_at_zero >= 2,
            "fixture lost its ties: {tied} tied rows, {censored_at_zero} gap curves censored at 0"
        );
        eprintln!(
            "vpc_rtte vs survfit: worst |ΔMCF| = {worst_mcf:.3e}, worst |ΔS| = {worst_km:.3e}"
        );
    }
}
