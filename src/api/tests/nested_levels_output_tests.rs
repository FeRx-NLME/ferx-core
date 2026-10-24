//! Post-fit output of a model with nested levels (#1082 PR4): the per-level κ̂, each κ's
//! shrinkage over its own level's groups (#1911's informed rule, per group), the typical
//! weight per level, the KAPPA rows `at LEVEL`, the sdtab κ̂ columns and `.fitrx`.

use crate::api::postfit::{kappa_weight_typicals, nested_kappa_facts};
use crate::parser::model_parser::parse_model_string;
use crate::stats::nested_levels::{assign_level_cells, NestedLayout};
use crate::types::{CompiledModel, Population, Subject};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// κ at STUDY and κ at ARM, compartment-free (so only a group's own observations inform
/// it). `extra_kappa` replaces the STUDY κ line.
fn model_with(study_kappa: &str) -> CompiledModel {
    parse_model_string(&format!(
        "[parameters]\n  theta TVB(10.0)\n  omega ETA ~ 0.09\n  level STUDY by STUDY in ID\n  \
         level ARM by ARM in STUDY\n  {study_kappa}\n  kappa K_ARM ~ 1.0 at ARM\n  \
         sigma ADD ~ 0.25\n[individual_parameters]\n  B = TVB * exp(ETA) + K_ST + K_ARM\n\
         [structural_model]\n  y = B\n[error_model]\n  DV ~ additive(ADD)\n"
    ))
    .expect("fixture parses")
}

fn model() -> CompiledModel {
    model_with("kappa K_ST ~ 4.0 at STUDY")
}

fn snap(study: f64, arm: f64) -> HashMap<String, f64> {
    [
        ("STUDY".to_string(), study),
        ("ARM".to_string(), arm),
        // A weight constant inside a study, different between studies.
        ("NS".to_string(), 10.0 * study),
    ]
    .into_iter()
    .collect()
}

/// Per subject: observations in (1,1) (1,1) (1,2) (2,1), and two `MDV = 1` rows, one in
/// arm (1,3) — an uninformed arm inside an informed study — and one in (3,1), the only
/// arm of study 3, which is therefore an uninformed **study**. Cells in group order:
/// (1,1) (1,2) (2,1) (1,3) (3,1).
fn population(n: usize) -> Population {
    let rows = [(1.0, 1.0), (1.0, 1.0), (1.0, 2.0), (2.0, 1.0)];
    let subjects = (1..=n)
        .map(|i| Subject {
            id: i.to_string(),
            obs_times: (0..rows.len()).map(|j| j as f64 + 1.0).collect(),
            observations: vec![10.0; rows.len()],
            obs_cmts: vec![1; rows.len()],
            cens: vec![0; rows.len()],
            covariates: snap(1.0, 1.0),
            obs_covariates: rows.iter().map(|&(s, a)| snap(s, a)).collect(),
            pk_only_times: vec![1.5, 2.5],
            pk_only_covariates: vec![snap(1.0, 3.0), snap(3.0, 1.0)],
            ..Default::default()
        })
        .collect();
    let mut p = Population {
        subjects,
        covariate_names: ["STUDY", "ARM", "NS"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        dv_column: "DV".to_string(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    assign_level_cells(&model(), &mut p);
    p
}

/// Per subject `i`, the per-cell κ̂ of a compact vector whose study κ̂ are
/// `[s1, s2, s3] + i` and arm κ̂ `[a1 … a5] + i`, as the inner loop would return them.
fn kappas(m: &CompiledModel, p: &Population) -> Vec<Vec<DVector<f64>>> {
    p.subjects
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let n = NestedLayout::of(m, s).expect("prepared");
            assert_eq!(n.n_compact(), 1 + 3 + 5, "η, 3 studies, 5 arms");
            let off = i as f64;
            let b = [
                0.0,
                0.5 + off,
                -0.3 + off,
                9.0 + off,
                0.1 + off,
                0.2 + off,
                0.3 + off,
                7.0 + off,
                8.0 + off,
            ];
            n.expand(&b).into_iter().map(DVector::from_vec).collect()
        })
        .collect()
}

fn omega_iov() -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 1.0]))
}

/// `1 − √mean(v²)/√var`, written out for the expected values.
fn shrink(vals: &[f64], var: f64) -> f64 {
    let ms = vals.iter().map(|v| v * v).sum::<f64>() / vals.len() as f64;
    1.0 - ms.sqrt() / var.sqrt()
}

#[test]
fn each_level_reports_each_group_s_kappa_once_with_its_labels() {
    let (m, p) = (model(), population(2));
    let facts = nested_kappa_facts(&m, &p, &kappas(&m, &p), &omega_iov()).expect("nested");
    assert_eq!(facts.kappa_levels, ["STUDY", "ARM"]);
    let [study, arm] = facts.by_level.as_slice() else {
        panic!("two levels")
    };
    assert_eq!(study.level, "STUDY");
    assert_eq!(study.columns, ["STUDY"]);
    assert_eq!(study.kappa_names, ["K_ST"]);
    assert_eq!(arm.columns, ["STUDY", "ARM"]);
    assert_eq!(arm.kappa_names, ["K_ARM"]);
    // STUDY: three groups per subject (not five cells), in first-record order.
    let s1: Vec<(Vec<f64>, Vec<f64>)> = study.groups[1]
        .iter()
        .map(|g| (g.labels.clone(), g.kappa.clone()))
        .collect();
    assert_eq!(
        s1,
        [
            (vec![1.0], vec![1.5]),
            (vec![2.0], vec![0.7]),
            (vec![3.0], vec![10.0])
        ]
    );
    // ARM: five groups, each named by its whole path.
    let a0: Vec<Vec<f64>> = arm.groups[0].iter().map(|g| g.labels.clone()).collect();
    assert_eq!(
        a0,
        [
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![1.0, 3.0],
            vec![3.0, 1.0]
        ]
    );
    assert_eq!(arm.groups[0][4].kappa, [8.0]);
}

/// #1911 per level: a group is informed when any of its cells is. Study 1 counts though
/// its arm 3 is uninformed; study 3, whose only arm is uninformed, does not. The two
/// halves fail on the two ways to get that wrong: pooling every group (study 3's 9.0,
/// 10.0 enter) and requiring every cell to be informed (study 1 drops out).
#[test]
fn shrinkage_pools_each_kappa_over_its_level_s_informed_groups() {
    let (m, p) = (model(), population(2));
    let facts = nested_kappa_facts(&m, &p, &kappas(&m, &p), &omega_iov()).unwrap();
    // K_ST: studies 1 and 2 of both subjects.
    let want_st = shrink(&[0.5, -0.3, 1.5, 0.7], 4.0);
    // K_ARM: arms (1,1) (1,2) (2,1) of both subjects.
    let want_arm = shrink(&[0.1, 0.2, 0.3, 1.1, 1.2, 1.3], 1.0);
    assert!(facts.shrinkage.iter().all(|s| s.is_finite()));
    assert!(
        (facts.shrinkage[0] - want_st).abs() < 1e-15,
        "level STUDY: {} vs {want_st}",
        facts.shrinkage[0]
    );
    assert!(
        (facts.shrinkage[1] - want_arm).abs() < 1e-15,
        "level ARM: {} vs {want_arm}",
        facts.shrinkage[1]
    );
    // The per-cell pool the one-level rule computes counts study 1 once per arm.
    let (cells, _) = crate::api::postfit::kappa_shrinkages(&m, &p, &kappas(&m, &p), &omega_iov());
    assert!(
        (cells[0] - want_st).abs() > 1e-3,
        "{} vs {want_st}",
        cells[0]
    );
}

#[test]
fn a_model_without_nested_levels_has_no_per_level_facts() {
    let one = parse_model_string(
        "[parameters]\n  theta TVB(10.0)\n  omega ETA ~ 0.09\n  kappa K ~ 1.0\n  \
         sigma ADD ~ 0.25\n[individual_parameters]\n  B = TVB * exp(ETA) + K\n\
         [structural_model]\n  y = B\n[error_model]\n  DV ~ additive(ADD)\n",
    )
    .unwrap();
    let k = DMatrix::from_element(1, 1, 1.0);
    assert!(nested_kappa_facts(&one, &population(1), &[], &k).is_none());
}

/// A weighted κ at STUDY takes one weight per study. Study 1 has three arms (NS = 10),
/// studies 2 and 3 one each (NS = 20, 30): over two subjects the per-study median is 20,
/// while the per-cell median — the one-level rule, one weight per arm — is 10.
#[test]
fn the_typical_weight_is_the_median_over_the_kappa_s_level_groups() {
    let m = model_with("kappa K_ST ~ 4.0 at STUDY weight = NS");
    let rows = [(1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (2.0, 1.0), (3.0, 1.0)];
    let mut p = population(2);
    for s in &mut p.subjects {
        s.obs_times = (0..rows.len()).map(|j| j as f64 + 1.0).collect();
        s.observations = vec![10.0; rows.len()];
        s.obs_cmts = vec![1; rows.len()];
        s.cens = vec![0; rows.len()];
        s.obs_covariates = rows.iter().map(|&(st, a)| snap(st, a)).collect();
        s.pk_only_times.clear();
        s.pk_only_covariates.clear();
    }
    assign_level_cells(&m, &mut p);
    let typical = kappa_weight_typicals(&m, &p, &m.default_params.theta);
    assert_eq!(typical, vec![Some(20.0), None]);
}

/// A group with no observation takes its weight from its own record (review r1 #5):
/// study 3 of [`population`] has only an `MDV = 1` row, with NS = 30. Per study the
/// weights are 10, 20, 30, so the median over two subjects is 20; reading study 3's
/// weight off the subject's covariates (NS = 10, study 1's) would give 10.
#[test]
fn an_unobserved_group_takes_its_weight_from_its_own_record() {
    let m = model_with("kappa K_ST ~ 4.0 at STUDY weight = NS");
    let mut p = population(2);
    assign_level_cells(&m, &mut p);
    let typical = kappa_weight_typicals(&m, &p, &m.default_params.theta);
    assert_eq!(typical, vec![Some(20.0), None]);
}

// ── the KAPPA rows, summary and YAML `at LEVEL` ──────────────────────────────────

/// A fit result with two kappas, `K_ST` (log-normal) and `K_ARM` (weight `NARM`, typical
/// arm 200 when `weighted`), with nested levels STUDY / ARM or none, FIX on `K_ST`, and
/// the covariance status.
fn kappa_result(
    nested: bool,
    fix: bool,
    weighted: bool,
    status: crate::types::CovarianceStatus,
) -> crate::types::FitResult {
    let mut r = crate::types::test_helpers::minimal_fit_result();
    r.omega_iov = Some(DMatrix::from_diagonal(&DVector::from_vec(vec![0.09, 4.0])));
    r.kappa_names = vec!["K_ST".into(), "K_ARM".into()];
    r.kappa_fixed = vec![fix, false];
    r.se_kappa = Some(vec![0.01, 1.5]);
    r.kappa_param_types = vec![
        crate::types::EtaParamType::LogNormal,
        crate::types::EtaParamType::LogNormal,
    ];
    if weighted {
        r.kappa_weights = vec![None, Some("NARM".into())];
        r.kappa_weight_typical = vec![None, Some(200.0)];
    }
    if nested {
        r.kappa_levels = vec!["STUDY".into(), "ARM".into()];
    }
    r.shrinkage_kappa = vec![0.1, 0.2];
    r.covariance_status = status;
    r
}

/// `(label, rest)` of every row line: the label is the 20-wide column after the indent,
/// up to the ` = ` that follows it; a weight line leaves that column blank and has an
/// empty label.
fn split_rows(rows: &str) -> Vec<(String, String)> {
    rows.lines()
        .map(|l| {
            if l.get(2..22).is_some_and(|c| c.trim().is_empty()) {
                return (String::new(), l.trim().to_string());
            }
            let (label, rest) = l.split_once(" = ").expect("a row line");
            (label.trim().to_string(), rest.to_string())
        })
        .collect()
}

/// The KAPPA rows' input space, every cell: levels {none, STUDY/ARM} × FIX × weighted ×
/// covariance {computed, failed}. The level is part of the label and nothing else:
/// `K_ST at STUDY`, `K_ST at STUDY [FIX]`, `K_ARM at ARM`, and every other character of
/// a nested row — note, SE, weight line — is the unnested row's. Both sides of the gate
/// in one test, so a level printed for a one-level fit and a level dropped from a
/// nested one each redden it.
#[test]
fn a_kappa_row_names_its_level_in_the_label_and_nowhere_else() {
    use crate::io::output::{format_kappa_rows_from, KappaRowsInput, KappaRowsOptions};
    use crate::types::CovarianceStatus;
    let rows = |r: &crate::types::FitResult| {
        format_kappa_rows_from(
            &KappaRowsInput::from_result(r).expect("IOV"),
            KappaRowsOptions::default(),
        )
    };
    let labels = |r: &[(String, String)]| -> Vec<String> {
        r.iter()
            .filter(|(l, _)| !l.is_empty())
            .map(|(l, _)| l.clone())
            .collect()
    };
    let rest =
        |r: &[(String, String)]| -> Vec<String> { r.iter().map(|(_, x)| x.clone()).collect() };
    let mut cells = 0;
    for fix in [false, true] {
        for weighted in [false, true] {
            for status in [CovarianceStatus::Computed, CovarianceStatus::Failed] {
                let cell = format!("fix={fix} weighted={weighted} {status:?}");
                let flat = split_rows(&rows(&kappa_result(false, fix, weighted, status.clone())));
                let nested = split_rows(&rows(&kappa_result(true, fix, weighted, status.clone())));
                let fixed = if fix { " [FIX]" } else { "" };
                assert_eq!(
                    labels(&flat),
                    [format!("K_ST{fixed}"), "K_ARM".to_string()],
                    "no level without nested levels, {cell}"
                );
                assert_eq!(
                    labels(&nested),
                    [format!("K_ST at STUDY{fixed}"), "K_ARM at ARM".to_string()],
                    "{cell}"
                );
                assert_eq!(rest(&nested), rest(&flat), "only the label moves, {cell}");
                assert_eq!(
                    nested.len(),
                    2 + usize::from(weighted),
                    "a weight line only when weighted, {cell}"
                );
                cells += 1;
            }
        }
    }
    assert_eq!(cells, 8);
}

/// The summary's pooled kappa shrinkage names each κ's level under nested levels (the
/// pool is that level's groups), and only then.
#[test]
fn the_summary_kappa_shrinkage_names_the_level_it_pooled_over() {
    use crate::types::CovarianceStatus;
    let summary = |nested| {
        crate::io::output::format_summary(&kappa_result(
            nested,
            false,
            false,
            CovarianceStatus::Computed,
        ))
    };
    let (flat, nested) = (summary(false), summary(true));
    assert!(
        flat.contains("  Kappa shrinkage: K_ST 10.0%, K_ARM 20.0%\n"),
        "{flat}"
    );
    assert!(
        nested.contains("  Kappa shrinkage: K_ST at STUDY 10.0%, K_ARM at ARM 20.0%\n"),
        "{nested}"
    );
}

/// The console's pooled kappa shrinkage block (#2083) names each κ's level under nested
/// levels and only then, agreeing with the summary's label. The whole block is pinned on
/// both sides of the gate, so a level dropped from a nested fit, a level printed for a
/// one-level fit, and any other change to a one-level fit's console output each redden
/// it. A NaN pooled entry and a per-occasion slot keep those arms in the fixture; the
/// slot lines keep the bare names, as the summary's do.
#[test]
fn the_console_kappa_shrinkage_names_the_level_it_pooled_over() {
    use crate::io::output::{format_kappa_shrinkage_lines, format_summary};
    use crate::types::CovarianceStatus;
    let result = |nested| {
        let mut r = kappa_result(nested, false, false, CovarianceStatus::Computed);
        r.shrinkage_kappa_by_occ = vec![vec![0.3, f64::NAN]];
        r
    };
    let (flat, nested) = (
        format_kappa_shrinkage_lines(&result(false)),
        format_kappa_shrinkage_lines(&result(true)),
    );
    assert_eq!(
        flat,
        [
            "\n--- Kappa Shrinkage (pooled) ---",
            "  K_ST shrinkage: 10.0%",
            "  K_ARM shrinkage: 20.0%",
            "  Per-occasion:",
            "    Occasion slot 1: K_ST 30.0%, K_ARM NaN",
        ]
    );
    assert_eq!(
        nested,
        [
            "\n--- Kappa Shrinkage (pooled) ---",
            "  K_ST at STUDY shrinkage: 10.0%",
            "  K_ARM at ARM shrinkage: 20.0%",
            "  Per-occasion:",
            "    Occasion slot 1: K_ST 30.0%, K_ARM NaN",
        ]
    );
    // The console and the summary agree on the label, on both sides of the gate.
    for (nested, lines) in [(false, &flat), (true, &nested)] {
        let summary = format_summary(&result(nested));
        for (k, line) in lines[1..3].iter().enumerate() {
            let label = line.trim().split(" shrinkage:").next().unwrap();
            let pct = (k + 1) * 10;
            assert!(
                summary.contains(&format!("{label} {pct}.0%")),
                "nested={nested}: {label:?} not in\n{summary}"
            );
        }
    }
    // A NaN pooled entry prints as such, with its level.
    let mut r = result(true);
    r.shrinkage_kappa[1] = f64::NAN;
    assert_eq!(
        format_kappa_shrinkage_lines(&r)[2],
        "  K_ARM at ARM shrinkage: NaN"
    );
    // No kappa shrinkage, no block.
    let mut r = result(true);
    r.shrinkage_kappa.clear();
    assert!(format_kappa_shrinkage_lines(&r).is_empty());
}

/// The fit YAML's `omega_iov:` entry of a nested κ says its level; a one-level fit's has
/// no `level:` line, so its YAML is unchanged.
#[test]
fn the_fit_yaml_records_each_kappa_s_level() {
    use crate::types::CovarianceStatus;
    let dir = std::env::temp_dir().join(format!("ferx-1082-pr4-yaml-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let yaml = |nested: bool| {
        let path = dir.join(format!("fit-{nested}.yaml"));
        let r = kappa_result(nested, false, false, CovarianceStatus::Computed);
        crate::io::output::write_estimates_yaml(&r, path.to_str().unwrap()).unwrap();
        std::fs::read_to_string(&path).unwrap()
    };
    let (flat, nested) = (yaml(false), yaml(true));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!flat.contains("level:"), "{flat}");
    assert!(
        nested.contains("  K_ST:\n    level: \"STUDY\"\n    variance: 0.090000\n"),
        "{nested}"
    );
    assert!(
        nested.contains("  K_ARM:\n    level: \"ARM\"\n    variance: 4.000000\n"),
        "{nested}"
    );
}
