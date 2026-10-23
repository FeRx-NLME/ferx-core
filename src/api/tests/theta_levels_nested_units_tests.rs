//! A θ level block against kappas at nested levels (#2034). A kappa's unit is a group
//! of its own level within one subject (`Unit::Level`), keyed by label path as
//! `stats::nested_levels` keys a group; the span certificate moves it on every cell of
//! that group. Binder-only fixtures: FD of the f64 predictor, no fit.

use super::absorption::{kshape, occ_pop, try_bind};
use super::readout_share::{EMAXY, T6};
use super::*;
use crate::parser::model_parser::LevelContrast;

/// `T6` with 0.5 in place of 0, so no `TIME`-keyed level is dead.
const T: [f64; 6] = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0];

/// Three sponsors (the subjects), two studies each, two arms per study, every arm on
/// the grid `T`. `STUDY` is unique across the population (the issue's design, studies
/// paired under a sponsor `ID`); `ARM` repeats in every study, so an arm is identified
/// by its path. `ARMID` and `SUBJ` number an arm and a sponsor; `W` is the record
/// index, which varies within every arm.
fn sponsor_pop() -> Population {
    let mut pop = population(3, 0);
    for (s, subject) in pop.subjects.iter_mut().enumerate() {
        let mut rows = Vec::new();
        for study in 1..=2 {
            for arm in 1..=2 {
                for &t in &T {
                    rows.push((t, study, arm));
                }
            }
        }
        // Time order, the arms of one study interleaved at each time.
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        let snap = |j: usize, study: usize, arm: usize| -> HashMap<String, f64> {
            let st = (10 * (s + 1) + study) as f64;
            HashMap::from([
                ("STUDY".to_string(), st),
                ("ARM".to_string(), arm as f64),
                ("ARMID".to_string(), 10.0 * st + arm as f64),
                ("SUBJ".to_string(), (s + 1) as f64),
                ("W".to_string(), j as f64),
            ])
        };
        let n = rows.len();
        subject.doses.clear();
        subject.covariates = snap(0, rows[0].1, rows[0].2);
        subject.obs_times = rows.iter().map(|r| r.0).collect();
        subject.obs_covariates = rows
            .iter()
            .enumerate()
            .map(|(j, r)| snap(j, r.1, r.2))
            .collect();
        subject.observations = vec![1.0; n];
        subject.obs_cmts = vec![1; n];
        subject.cens = vec![0; n];
    }
    pop
}

/// A compartment-free placebo/Emax model with a κ per study and one per arm: `e0` is
/// the `E0` line, `cols` the block's columns, `contrast` its contrast (`""` = auto).
/// The random effects `e0` does not read sit on `Z`, which `y` never reads.
fn nested_model(e0: &str, cols: &str, contrast: &str) -> String {
    let modifier = if contrast.is_empty() {
        String::new()
    } else {
        format!(", contrast = {contrast}")
    };
    format!(
        r#"
[parameters]
  theta TVE0(1.5, -10.0, 10.0)
  theta PLACEBO[{cols}{modifier}](0.0, -10.0, 10.0)
  theta TVEMAX(3.0, 0.1, 20.0)
  theta TVET50(1.5, 0.1, 20.0)
  omega ETA_E0 ~ 0.1
  level STUDY by STUDY in ID
  level ARM by ARM in STUDY
  kappa K_ST ~ 0.1 at STUDY
  kappa K_ARM ~ 0.1 at ARM
  sigma ADD ~ 0.1
[individual_parameters]
  EMAX = TVEMAX
  ET50 = TVET50
{e0}
  Z = TVE0 * exp(ETA_E0 + K_ST + K_ARM)
[structural_model]
  y = E0 + {EMAXY}
[error_model]
  DV ~ additive(ADD)
"#
    )
}

/// The issue's case: a block keyed on the time column, nested in the studies, beside a
/// kappa at `STUDY` — the binder measures the kappa on the studies, so the block is
/// absorbed and auto takes the within-study contrast, where an explicit global
/// contrast is refused naming the study.
///
/// Mutation — `Unit::Level` read one level too deep for `STUDY` (or as the cell, its
/// occasion): `[STUDY, TIME]` holds both arms of a study, so it does not nest in arms,
/// the kappa is not absorbed and auto binds global.
#[test]
fn a_time_keyed_block_nested_in_the_studies_absorbs_the_study_kappa() {
    let pop = sponsor_pop();
    let text = |c: &str| nested_model("  E0 = TVE0 + PLACEBO + K_ST", "STUDY, TIME", c);
    assert_eq!(
        try_bind(&text(""), &pop).map(|(c, _)| c),
        Ok(LevelContrast::SumToZeroWithin),
        "auto"
    );
    let err = try_bind(&text("sum_to_zero"), &pop).expect_err("global");
    for must in [
        "theta PLACEBO[STUDY, TIME]: `contrast = sum_to_zero` lets the levels reproduce \
         `K_ST` at every `STUDY` group",
        "the two are the same quantity, so the model is not identified",
        "Use `contrast = sum_to_zero_within`",
    ] {
        assert!(err.contains(must), "lacks {must:?}: {err}");
    }
    assert!(!err.contains("occasion"), "{err}");
}

/// Per level, the differential pair that straddles its unit: a block nested in the
/// kappa's groups absorbs it (auto goes within), one that crosses them does not (auto
/// stays global). Each row names the level whose unit it pins.
///
/// - `STUDY`, crossing by subject: `[SUBJ, TIME]` holds both studies of a sponsor.
///   Mutation — the study kappa measured on the subject (`Unit::Subject`): it nests,
///   and binds within.
/// - `STUDY`, nested: `[STUDY, TIME]`. Mutation — measured on the arm: binds global.
/// - `ARM`, crossing: `[STUDY, TIME]` holds both arms of a study. Mutation — the arm
///   kappa measured on its study (`Unit::Level(0)`): it nests, and binds within.
/// - `ARM`, nested: `[ARMID, TIME]`. Mutation — measured on the subject while the
///   levels cross it is not reachable (every arm is one sponsor's), so the row holds the
///   absorbed side of the gate.
///
/// The arm's cell is its occasion, so `Unit::Occasion` for the arm kappa is an
/// equivalent mutation, not killed here.
#[test]
fn each_level_measures_its_kappa_on_its_own_groups() {
    let pop = sponsor_pop();
    let rows = [
        ("STUDY", "K_ST", "SUBJ, TIME", LevelContrast::SumToZero),
        (
            "STUDY",
            "K_ST",
            "STUDY, TIME",
            LevelContrast::SumToZeroWithin,
        ),
        ("ARM", "K_ARM", "STUDY, TIME", LevelContrast::SumToZero),
        (
            "ARM",
            "K_ARM",
            "ARMID, TIME",
            LevelContrast::SumToZeroWithin,
        ),
    ];
    let mut wrong = Vec::new();
    for (level, kappa, cols, want) in rows {
        let text = nested_model(&format!("  E0 = TVE0 + PLACEBO + {kappa}"), cols, "");
        let got = try_bind(&text, &pop).map(|(c, _)| c);
        if got != Ok(want) {
            wrong.push(format!(
                "level {level}: {kappa} on [{cols}]: {got:?}, want {want:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The span certificate on a level: the study kappa in proportional branches on `W`,
/// which varies within every arm, so no expression reads both the block and the kappa
/// and the walk names no site. The kappa moves every arm of its study, and so does a
/// `[STUDY, TIME]` level, record by record: the certificate measures it absorbed, auto
/// goes within, and an explicit global contrast is refused with the measured clause
/// naming the study.
///
/// Mutation — the kappa moved on one cell of its group only (`unit_groups` takes the
/// first cell): its column covers one arm, a level's both, so it is not in their span
/// and the block binds global.
#[test]
fn the_span_certificate_moves_a_study_kappa_on_every_arm() {
    let pop = sponsor_pop();
    let k = "TVE0 + PLACEBO + K_ST";
    let e0 = format!("  if (W > 5) {{\n    E0 = {k}\n  }} else {{\n    E0 = 2 * ({k})\n  }}");
    let text = |c: &str| nested_model(&e0, "STUDY, TIME", c);
    assert_eq!(
        try_bind(&text(""), &pop).map(|(c, _)| c),
        Ok(LevelContrast::SumToZeroWithin),
        "auto"
    );
    let err = try_bind(&text("sum_to_zero"), &pop).expect_err("global");
    for must in [
        "the random effect `K_ST` moves `y`, at every record of each `STUDY` group, exactly \
         as some combination of this block's levels does",
        "reproduce `K_ST` at every `STUDY` group",
    ] {
        assert!(err.contains(must), "lacks {must:?}: {err}");
    }
    // The twin crosses the studies: the kappa is not absorbed, and auto is global.
    assert_eq!(
        try_bind(&nested_model(&e0, "SUBJ, TIME", ""), &pop).map(|(c, _)| c),
        Ok(LevelContrast::SumToZero),
        "[SUBJ, TIME]"
    );
}

/// A one-level model and its explicit two-level spelling, a singleton outer level
/// (`SUBJ` by a constant `ONE`) above the occasions, give the binder the same verdict
/// over the #1678 kappa grid: the same contrast and free count, or the same refusal
/// with the occasion called the `OCC` group.
///
/// Mutation — the nested spelling's kappa measured on the outer level: the period and
/// `[STUDY, OCC]` cells absorb it on the subject and disagree.
#[test]
fn a_one_level_model_and_its_two_level_spelling_bind_alike() {
    let st = "STUDY, TIME";
    let so = "STUDY, OCC";
    let cells: Vec<(&str, &str, bool, Population)> = vec![
        ("K-E0", "STUDY", false, occ_pop(3, 1, &T6, 2, false)),
        ("K-E0", "STUDY", true, occ_pop(3, 1, &T6, 3, true)),
        ("K-E0", st, false, occ_pop(3, 1, &T6, 2, false)),
        ("K-E0", st, true, occ_pop(3, 1, &T6, 2, true)),
        ("K-E0", so, false, occ_pop(3, 1, &T6, 2, false)),
        ("K-E0", so, true, occ_pop(3, 1, &T6, 3, true)),
        ("K-E0", so, false, occ_pop(3, 2, &T6, 2, false)),
        ("K-y", st, false, occ_pop(3, 1, &T6, 2, false)),
        ("K-EMAX", st, false, occ_pop(3, 1, &T6, 3, false)),
        ("K-unread", st, true, occ_pop(3, 1, &T6, 2, true)),
    ];
    let two_level = |text: String| {
        let out = text.replace(
            "  kappa KAPPA_E0 ~ 0.1\n",
            "  level SUBJ by ONE in ID\n  level OCC by OCC in SUBJ\n  kappa KAPPA_E0 ~ 0.1 at OCC\n",
        );
        assert_ne!(out, text, "the nested spelling declares the levels");
        out
    };
    let mut wrong = Vec::new();
    let mut refused = 0;
    for (shape, cols, eta, pop) in &cells {
        let mut nested_pop = pop.clone();
        for s in nested_pop.subjects.iter_mut() {
            s.doses.clear();
            for m in s.obs_covariates.iter_mut() {
                m.insert("ONE".to_string(), 1.0);
            }
        }
        let mut one_pop = pop.clone();
        for s in one_pop.subjects.iter_mut() {
            s.doses.clear();
        }
        for c in ["", "sum_to_zero", "sum_to_zero_within", "ref", "none"] {
            let tag = format!("{shape} η={eta} [{cols}] {c:?}");
            let one = try_bind(&kshape(shape, *eta, cols, c), &one_pop);
            let two = try_bind(&two_level(kshape(shape, *eta, cols, c)), &nested_pop)
                .map_err(|e| e.replace("`OCC` group", "occasion"));
            refused += usize::from(one.is_err());
            if one != two {
                wrong.push(format!("{tag}:\n  one level {one:?}\n  two levels {two:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    // Both sides of the gate: some cells refused, some bound.
    assert!(
        (1..cells.len() * 5).contains(&refused),
        "{refused} of {} refused",
        cells.len() * 5
    );
}

/// Two kappas at two nested levels absorbed together (#2034 review r1). On
/// `[ARMID, TIME]` both `K_ST` and `K_ARM` on `E0` are absorbed, and the within-arm
/// contrast parts them: each arm lies inside one study and is its own arm, so within
/// leaves every arm's mean to the kappas, which the replication of arms in studies
/// and of studies in sponsors tells apart. Auto goes within, and an explicit global
/// contrast is refused, naming a kappa it lets the levels reproduce.
///
/// The two sides of the gate, each a refusal of the joint clause:
/// - `[STUDY, ARMID]` groups by study, and the arm kappa varies inside a study.
///   Mutation — drop the nesting condition of `within_parts_them`: it binds.
/// - one arm per study: the two kappas group the records alike, so within leaves the
///   two one mean per study. Mutation — drop the apartness condition: it binds.
///
/// Mutation — `within_parts_them` always false: `[ARMID, TIME]` is refused.
#[test]
fn kappas_at_two_levels_absorbed_together_are_parted_by_the_within_contrast() {
    let pop = sponsor_pop();
    let e0 = "  E0 = TVE0 + PLACEBO + K_ST + K_ARM";
    let text = |cols: &str, c: &str| nested_model(e0, cols, c);
    assert_eq!(
        try_bind(&text("ARMID, TIME", ""), &pop).map(|(c, _)| c),
        Ok(LevelContrast::SumToZeroWithin),
        "[ARMID, TIME] auto"
    );
    let err = try_bind(&text("ARMID, TIME", "sum_to_zero"), &pop).expect_err("global");
    assert!(
        err.contains(
            "`contrast = sum_to_zero` lets the levels reproduce `K_ST` at every `STUDY` group"
        ),
        "{err}"
    );
    let joint = "the levels absorb `K_ST` and `K_ARM` together";
    let err = try_bind(&text("STUDY, ARMID", ""), &pop).expect_err("[STUDY, ARMID]");
    assert!(err.contains(joint), "[STUDY, ARMID]: {err}");
    let mut one_arm = pop.clone();
    for s in one_arm.subjects.iter_mut() {
        let keep: Vec<usize> = (0..s.obs_times.len())
            .filter(|&j| s.obs_covariates[j]["ARM"] == 1.0)
            .collect();
        s.obs_times = keep.iter().map(|&j| s.obs_times[j]).collect();
        s.obs_covariates = keep.iter().map(|&j| s.obs_covariates[j].clone()).collect();
        s.observations = vec![1.0; keep.len()];
        s.obs_cmts = vec![1; keep.len()];
        s.cens = vec![0; keep.len()];
    }
    let err = try_bind(&text("ARMID, TIME", ""), &one_arm).expect_err("one arm per study");
    assert!(err.contains(joint), "one arm per study: {err}");
}
