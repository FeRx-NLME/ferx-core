use super::*;
use crate::types::{MuTransform, Population};
use nalgebra::DMatrix;
use std::io::Write;

const ADDITIVE_MODEL: &str = r"
[parameters]
  theta TVCL(150.0, 0.0, 1000.0)
  theta TH_CRCL(2.0, 0.0, 50.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.2
  omega ETA_V ~ 0.1
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = (TVCL + (CRCL - 90.0) * TH_CRCL) * exp(ETA_CL)
  V  = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

fn model(src: &str) -> CompiledModel {
    crate::parser::model_parser::parse_model_string(src).expect("model parses")
}

/// One subject per CRCL value: a dose at 0 and one observation at 1 h.
fn pop_with_crcl(crcl: &[f64]) -> Population {
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,CRCL\n");
    for (i, c) in crcl.iter().enumerate() {
        let id = i + 1;
        csv.push_str(&format!("{id},0,0,100,1,1,{c}\n{id},1,5.0,0,0,1,{c}\n"));
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    crate::io::datareader::read_nonmem_csv(f.path(), Some(&["CRCL"]), None).unwrap()
}

/// Two-record subject whose CRCL changes between the dose and the sample.
fn pop_time_varying() -> Population {
    let csv = "ID,TIME,DV,AMT,EVID,CMT,CRCL\n1,0,0,100,1,1,60\n1,1,5.0,0,0,1,80\n2,0,0,100,1,1,100\n2,1,4.0,0,0,1,100\n";
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    crate::io::datareader::read_nonmem_csv(f.path(), Some(&["CRCL"]), None).unwrap()
}

const CRCL_GRID: [f64; 12] = [
    40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
];

fn diag_omega(vars: &[f64]) -> DMatrix<f64> {
    DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(vars))
}

fn only_group<'m>(
    m: &'m CompiledModel,
    pop: &Population,
    omega: &DMatrix<f64>,
) -> CovariateMuGroup<'m> {
    let fixed = vec![false; m.theta_names.len()];
    let (mut groups, notes) = resolve_covariate_mu_groups(m, pop, &[], &fixed, omega);
    assert!(notes.is_empty(), "unexpected notes: {notes:?}");
    assert_eq!(groups.len(), 1, "exactly one group expected");
    groups.remove(0)
}

/// Etas that place every subject's φ exactly at the planted typical value:
/// `η_i = g(A_i(θ_true)) − g(A_i(θ_start))`, so the exact solver's optimum *is*
/// θ_true and the test has a closed-form answer, not a tolerance story.
fn planted_etas(
    g: &CovariateMuGroup<'_>,
    pop: &Population,
    start: &[f64],
    truth: &[f64],
) -> Vec<Vec<f64>> {
    let mu_s = g.mus(start, pop);
    let mu_t = g.mus(truth, pop);
    (0..pop.subjects.len())
        .map(|i| vec![mu_t[i] - mu_s[i], 0.0])
        .collect()
}

/// [`ADDITIVE_MODEL`] with `TH_CRCL` also feeding bioavailability, so the
/// observation likelihood still depends on it once `φ_CL` is preserved.
const SHARED_THETA_MODEL: &str = r"
[parameters]
  theta TVCL(150.0, 0.0, 1000.0)
  theta TH_CRCL(2.0, 0.0, 50.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.2
  omega ETA_V ~ 0.1
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = (TVCL + (CRCL - 90.0) * TH_CRCL) * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  FR = TH_CRCL * 0.01

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=1.0, f=FR)

[error_model]
  DV ~ proportional(EPS)
";

/// The exact engine drops the data term. That is only legitimate when the
/// group's thetas reach the data through this typical value alone — here
/// `TH_CRCL` also sets `FR`, so the group must keep the term instead.
/// [`additive_group_is_detected_and_time_constant`] is the control: the same
/// covariate model with an unshared `TH_CRCL` does take the exact engine.
#[test]
fn a_theta_the_rest_of_the_model_reads_keeps_the_data_term() {
    let m = model(SHARED_THETA_MODEL);
    assert_eq!(m.covariate_mu_refs.len(), 1);
    assert_eq!(m.covariate_mu_refs[0].shared_thetas, vec!["TH_CRCL"]);

    let pop = pop_with_crcl(&CRCL_GRID);
    let omega = diag_omega(&[0.2, 0.1]);
    let free = vec![false; m.theta_names.len()];
    let (groups, notes) = resolve_covariate_mu_groups(&m, &pop, &[], &free, &omega);
    assert_eq!(
        groups.len(),
        1,
        "the group is still estimated, just differently"
    );
    assert!(groups[0].needs_data_term);
    assert!(
        notes.iter().any(|n| n.contains("TH_CRCL")),
        "the user is told which theta forced it: {notes:?}"
    );

    // A FIXed theta never moves, so no term can be mis-maximised in it and the
    // exact engine stays available. Without this the check above is satisfied by
    // routing on the mere presence of a name in `shared_thetas`.
    let mut fixed = vec![false; m.theta_names.len()];
    fixed[m
        .theta_names
        .iter()
        .position(|t| t == "TH_CRCL")
        .expect("TH_CRCL is declared")] = true;
    let (groups, _) = resolve_covariate_mu_groups(&m, &pop, &[], &fixed, &omega);
    assert_eq!(groups.len(), 1);
    assert!(!groups[0].needs_data_term);
}

#[test]
fn additive_group_is_detected_and_time_constant() {
    let m = model(ADDITIVE_MODEL);
    assert_eq!(m.covariate_mu_refs.len(), 1);
    let spec = &m.covariate_mu_refs[0];
    assert_eq!(spec.eta_name, "ETA_CL");
    assert_eq!(spec.theta_names, vec!["TVCL", "TH_CRCL"]);
    assert_eq!(spec.transform, MuTransform::Log);
    assert_eq!(spec.covariate_names, vec!["CRCL"]);
    // The additive form matches no single-anchor pattern, so `mu_refs` is
    // exactly what it was before #619: V only.
    assert!(!m.mu_refs.contains_key("ETA_CL"));
    assert!(m.mu_refs.contains_key("ETA_V"));

    let pop = pop_with_crcl(&CRCL_GRID);
    let g = only_group(&m, &pop, &diag_omega(&[0.2, 0.1]));
    assert_eq!(g.eta_idx, 0);
    assert_eq!(g.theta_idx, vec![0, 1]);
    assert!(!g.needs_data_term);
}

#[test]
fn mu_is_log_of_the_typical_value_per_subject() {
    let m = model(ADDITIVE_MODEL);
    let pop = pop_with_crcl(&[40.0, 140.0]);
    let g = only_group(&m, &pop, &diag_omega(&[0.2, 0.1]));
    let theta = [120.0, 1.4, 10.0];
    let mus = g.mus(&theta, &pop);
    assert!((mus[0] - (120.0 + (40.0 - 90.0) * 1.4f64).ln()).abs() < 1e-12);
    assert!((mus[1] - (120.0 + (140.0 - 90.0) * 1.4f64).ln()).abs() < 1e-12);
    // A non-positive lognormal typical value is inadmissible, not log(1e-30).
    assert!(g.mu(&[10.0, 2.0, 10.0], &pop.subjects[0]).is_nan());
}

#[test]
fn exact_solver_recovers_planted_thetas() {
    let m = model(ADDITIVE_MODEL);
    let pop = pop_with_crcl(&CRCL_GRID);
    let omega = diag_omega(&[0.2, 0.1]);
    let g = only_group(&m, &pop, &omega);
    let start = [150.0, 2.0, 10.0];
    let truth = [120.0, 1.4, 10.0];
    let etas = planted_etas(&g, &pop, &start, &truth);
    let fixed = [false, false, false];
    let input = GroupStepInput {
        theta: &start,
        theta_lower: &m.default_params.theta_lower,
        theta_upper: &m.default_params.theta_upper,
        theta_fixed: &fixed,
        theta_packs_log: &[true, true, true],
        omega: &omega,
        etas: &etas,
    };
    let out = g.solve_exact(&pop, &input).expect("solvable");
    assert!((out[0] - 120.0).abs() < 1e-6, "TVCL = {}", out[0]);
    assert!((out[1] - 1.4).abs() < 1e-8, "TH_CRCL = {}", out[1]);
    assert_eq!(out[2], 10.0, "a theta outside the group is untouched");
}

#[test]
fn exact_solver_leaves_a_fixed_theta_alone() {
    let m = model(ADDITIVE_MODEL);
    let pop = pop_with_crcl(&CRCL_GRID);
    let omega = diag_omega(&[0.2, 0.1]);
    let g = only_group(&m, &pop, &omega);
    let start = [150.0, 2.0, 10.0];
    let truth = [120.0, 1.4, 10.0];
    let etas = planted_etas(&g, &pop, &start, &truth);
    let fixed = [true, false, false];
    let input = GroupStepInput {
        theta: &start,
        theta_lower: &m.default_params.theta_lower,
        theta_upper: &m.default_params.theta_upper,
        theta_fixed: &fixed,
        theta_packs_log: &[true, true, true],
        omega: &omega,
        etas: &etas,
    };
    let out = g.solve_exact(&pop, &input).expect("solvable");
    assert_eq!(out[0], 150.0, "FIXed TVCL must not move");
    assert!(out[1] < 2.0, "the free slope compensates: {}", out[1]);
    // With every group theta FIXed there is nothing to solve.
    let all_fixed = [true, true, false];
    let input2 = GroupStepInput {
        theta_fixed: &all_fixed,
        ..input
    };
    assert!(g.solve_exact(&pop, &input2).is_none());
}

#[test]
fn exact_solver_clamps_to_the_declared_bounds() {
    let src = ADDITIVE_MODEL.replace(
        "theta TH_CRCL(2.0, 0.0, 50.0)",
        "theta TH_CRCL(0.5, 0.0, 1.0)",
    );
    let m = model(&src);
    let pop = pop_with_crcl(&CRCL_GRID);
    let omega = diag_omega(&[0.2, 0.1]);
    let g = only_group(&m, &pop, &omega);
    let start = [150.0, 0.5, 10.0];
    let truth = [120.0, 1.4, 10.0]; // slope truth above the upper bound 1.0
    let etas = planted_etas(&g, &pop, &start, &truth);
    let fixed = [false; 3];
    let input = GroupStepInput {
        theta: &start,
        theta_lower: &m.default_params.theta_lower,
        theta_upper: &m.default_params.theta_upper,
        theta_fixed: &fixed,
        theta_packs_log: &[true, true, true],
        omega: &omega,
        etas: &etas,
    };
    let out = g.solve_exact(&pop, &input).expect("solvable");
    assert!(
        out[1] <= 1.0 + 1e-12,
        "slope must respect the bound: {}",
        out[1]
    );
    assert!(out[1] > 0.99, "and sit at it: {}", out[1]);
}

#[test]
fn exact_solver_declines_an_inadmissible_start() {
    let m = model(ADDITIVE_MODEL);
    let pop = pop_with_crcl(&[40.0, 140.0]);
    let omega = diag_omega(&[0.2, 0.1]);
    let g = only_group(&m, &pop, &omega);
    // 10 + (40 − 90)·2 < 0: the subject's lognormal typical value is not positive.
    let start = [10.0, 2.0, 10.0];
    let etas = vec![vec![0.0, 0.0]; 2];
    let fixed = [false; 3];
    let input = GroupStepInput {
        theta: &start,
        theta_lower: &m.default_params.theta_lower,
        theta_upper: &m.default_params.theta_upper,
        theta_fixed: &fixed,
        theta_packs_log: &[true, true, true],
        omega: &omega,
        etas: &etas,
    };
    assert!(g.solve_exact(&pop, &input).is_none());
}

/// With a block Ω the other component's residual enters the target through
/// `W_kl / W_kk`. Plant etas so that the *corrected* target sits exactly at
/// θ_true: the solver must land there, and would not if it ignored the cross
/// term.
#[test]
fn exact_solver_folds_the_block_omega_cross_term() {
    let m = model(ADDITIVE_MODEL);
    let pop = pop_with_crcl(&CRCL_GRID);
    let omega = DMatrix::from_row_slice(2, 2, &[0.2, 0.08, 0.08, 0.1]);
    let g = only_group(&m, &pop, &omega);
    let w = omega.clone().try_inverse().unwrap();
    let c = w[(0, 1)] / w[(0, 0)];
    assert!(
        c.abs() > 0.1,
        "the fixture must have a live cross term: {c}"
    );
    let start = [150.0, 2.0, 10.0];
    let truth = [120.0, 1.4, 10.0];
    let mu_s = g.mus(&start, &pop);
    let mu_t = g.mus(&truth, &pop);
    let etas: Vec<Vec<f64>> = (0..pop.subjects.len())
        .map(|i| {
            let eta_v = 0.3 * ((i as f64) - 5.5); // a V residual with non-zero mean structure
            vec![mu_t[i] - mu_s[i] - c * eta_v, eta_v]
        })
        .collect();
    let fixed = [false; 3];
    let input = GroupStepInput {
        theta: &start,
        theta_lower: &m.default_params.theta_lower,
        theta_upper: &m.default_params.theta_upper,
        theta_fixed: &fixed,
        theta_packs_log: &[true, true, true],
        omega: &omega,
        etas: &etas,
    };
    let out = g.solve_exact(&pop, &input).expect("solvable");
    assert!((out[0] - 120.0).abs() < 1e-5, "TVCL = {}", out[0]);
    assert!((out[1] - 1.4).abs() < 1e-7, "TH_CRCL = {}", out[1]);

    // Control: the same etas under a *diagonal* Ω solve to a different point,
    // so the assertion above cannot be satisfied by ignoring the cross term.
    let diag = diag_omega(&[0.2, 0.1]);
    let g2 = only_group(&m, &pop, &diag);
    let input2 = GroupStepInput {
        omega: &diag,
        ..input
    };
    let out2 = g2.solve_exact(&pop, &input2).expect("solvable");
    assert!(
        (out2[1] - 1.4).abs() > 1e-3,
        "control must differ: {}",
        out2[1]
    );
}

/// The numerical engine's prior term must reproduce the exact engine when the
/// data term is constant — that is the check on its quadratic form.
#[test]
fn numerical_solver_matches_exact_when_the_data_term_is_constant() {
    let m = model(ADDITIVE_MODEL);
    let pop = pop_with_crcl(&CRCL_GRID);
    let omega = diag_omega(&[0.2, 0.1]);
    let g = only_group(&m, &pop, &omega);
    let start = [150.0, 2.0, 10.0];
    let truth = [120.0, 1.4, 10.0];
    let etas = planted_etas(&g, &pop, &start, &truth);
    let fixed = [false; 3];
    let input = GroupStepInput {
        theta: &start,
        theta_lower: &m.default_params.theta_lower,
        theta_upper: &m.default_params.theta_upper,
        theta_fixed: &fixed,
        theta_packs_log: &[true, true, true],
        omega: &omega,
        etas: &etas,
    };
    let out = g
        .solve_numerical(&pop, &input, 200, &|_theta, _shift| 0.0)
        .expect("solvable");
    assert!((out[0] - 120.0).abs() < 0.5, "TVCL = {}", out[0]);
    assert!((out[1] - 1.4).abs() < 0.01, "TH_CRCL = {}", out[1]);
}

#[test]
fn numerical_solver_passes_the_phi_preserving_shift_to_the_data_term() {
    let m = model(ADDITIVE_MODEL);
    let pop = pop_with_crcl(&CRCL_GRID);
    let omega = diag_omega(&[0.2, 0.1]);
    let g = only_group(&m, &pop, &omega);
    let start = [150.0, 2.0, 10.0];
    let etas = vec![vec![0.0, 0.0]; pop.subjects.len()];
    let fixed = [false; 3];
    let input = GroupStepInput {
        theta: &start,
        theta_lower: &m.default_params.theta_lower,
        theta_upper: &m.default_params.theta_upper,
        theta_fixed: &fixed,
        theta_packs_log: &[true, true, true],
        omega: &omega,
        etas: &etas,
    };
    let mu_old = g.mus(&start, &pop);
    let seen = std::cell::Cell::new(false);
    let pop_ref = &pop;
    let g_ref = &g;
    let data = |theta: &[f64], shift: &[f64]| -> f64 {
        let mu_new = g_ref.mus(theta, pop_ref);
        for i in 0..shift.len() {
            assert!((shift[i] - (mu_old[i] - mu_new[i])).abs() < 1e-12);
        }
        seen.set(true);
        0.0
    };
    g.solve_numerical(&pop, &input, 5, &data).expect("solvable");
    assert!(seen.get(), "the data term must have been evaluated");
}

#[test]
fn time_varying_covariate_keeps_the_data_term() {
    let m = model(ADDITIVE_MODEL);
    let pop = pop_time_varying();
    let g = only_group(&m, &pop, &diag_omega(&[0.2, 0.1]));
    assert!(g.needs_data_term);
}

#[test]
fn resolve_drops_a_group_sharing_a_theta_with_another_etas_anchor() {
    let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TH_WT(0.01, 0.0, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = (TVCL + TH_WT * CRCL) * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
    let m = model(src);
    assert_eq!(m.covariate_mu_refs.len(), 1, "the parser records the group");
    let pop = pop_with_crcl(&[60.0, 90.0]);
    let fixed = [false, false];
    // (TVCL, ETA_CL) is the single-anchor pair of the *other* eta.
    let (groups, notes) =
        resolve_covariate_mu_groups(&m, &pop, &[(0, 0)], &fixed, &diag_omega(&[0.09, 0.04]));
    assert!(groups.is_empty());
    assert_eq!(notes.len(), 1);
    assert!(
        notes[0].contains("TVCL") && notes[0].contains("ETA_CL"),
        "{}",
        notes[0]
    );
}

#[test]
fn resolve_drops_weak_iiv_and_all_fixed_groups() {
    let m = model(ADDITIVE_MODEL);
    let pop = pop_with_crcl(&[60.0, 90.0]);
    let (groups, notes) = resolve_covariate_mu_groups(
        &m,
        &pop,
        &[],
        &[false, false, false],
        &diag_omega(&[1e-4, 0.1]),
    );
    assert!(groups.is_empty());
    assert_eq!(notes.len(), 1);
    assert!(notes[0].contains("negligible variance"), "{}", notes[0]);

    let (groups, notes) = resolve_covariate_mu_groups(
        &m,
        &pop,
        &[],
        &[true, true, false],
        &diag_omega(&[0.2, 0.1]),
    );
    assert!(groups.is_empty());
    assert!(
        notes.is_empty(),
        "an all-FIX group is silently inert: {notes:?}"
    );
}

#[test]
fn logit_group_mu_is_the_typical_value_itself() {
    let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta LOGIT_F(0.4, -10.0, 10.0)
  theta TH_SEX(0.5, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  omega ETA_F ~ 0.04
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  F  = inv_logit(LOGIT_F + TH_SEX * CRCL + ETA_F)
  V  = 10.0 * F

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
    let m = model(src);
    assert_eq!(m.covariate_mu_refs.len(), 1);
    assert_eq!(m.covariate_mu_refs[0].transform, MuTransform::Logit);
    let pop = pop_with_crcl(&[1.0]);
    let g = only_group(&m, &pop, &diag_omega(&[0.09, 0.04]));
    assert_eq!(g.eta_idx, 1);
    let mu = g.mu(&[1.0, 0.4, 0.5], &pop.subjects[0]);
    assert!(
        (mu - 0.9).abs() < 1e-12,
        "logit-scale mu = LOGIT_F + TH_SEX·CRCL: {mu}"
    );
}

// ── #918 review follow-ups ───────────────────────────────────────────────────

/// [`ADDITIVE_MODEL`] with a second typical value reading the *same* eta. The
/// exact engine drops the observation term on the argument that freezing `φ_CL`
/// freezes the individual parameter — but the step re-centres `η_CL`, and that
/// moves `V` as well, so the data term is live in `TVCL`/`TH_CRCL` after all.
///
/// `retire_superseded_group` does not catch it: it is keyed on
/// `split_typical_and_eta`, which returns `None` for `TVV * exp(0.5 * ETA_CL)`
/// (the exp factor must be exactly `exp(ETA)`), so nothing is retired.
const SHARED_ETA_MODEL: &str = r"
[parameters]
  theta TVCL(150.0, 0.0, 1000.0)
  theta TH_CRCL(2.0, 0.0, 50.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.2
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = (TVCL + (CRCL - 90.0) * TH_CRCL) * exp(ETA_CL)
  V  = TVV * exp(0.5 * ETA_CL)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// A second typical value reading the group's eta forces the prior-plus-data
/// engine. Control: [`ADDITIVE_MODEL`], whose `V` carries its own `ETA_V`, is
/// still exact — so the flag tracks eta sharing, not merely "the model has two
/// individual parameters".
#[test]
fn an_eta_a_second_typical_value_reads_keeps_the_data_term() {
    let shared = model(SHARED_ETA_MODEL);
    assert_eq!(shared.covariate_mu_refs.len(), 1, "the group survives");
    assert!(
        shared.covariate_mu_refs[0].eta_shared,
        "ETA_CL is read by V as well"
    );

    let pop = pop_with_crcl(&CRCL_GRID);
    let free = vec![false; shared.theta_names.len()];
    let (groups, notes) =
        resolve_covariate_mu_groups(&shared, &pop, &[], &free, &diag_omega(&[0.2]));
    assert_eq!(groups.len(), 1, "estimated, just by the other engine");
    assert!(
        groups[0].needs_data_term,
        "re-centring ETA_CL moves V, so the data term is not constant"
    );
    assert!(
        notes.iter().any(|n| n.contains("ETA_CL")),
        "the user is told which eta forced it: {notes:?}"
    );

    let control = model(ADDITIVE_MODEL);
    assert!(
        !control.covariate_mu_refs[0].eta_shared,
        "V has its own ETA_V there"
    );
    let g = only_group(&control, &pop, &diag_omega(&[0.2, 0.1]));
    assert!(!g.needs_data_term, "control still takes the exact engine");
}

/// `can_step` is the predicate IMP/IMPMAP must ask *before* pinning a group's
/// thetas out of the weighted M-step, so it has to agree exactly with the
/// condition under which both engines return `None`. Asserted on both arms of
/// the additive model: a θ whose typical value stays positive for every subject
/// versus one that goes negative for the lowest-CRCL subject.
#[test]
fn can_step_agrees_with_both_engines_returning_none() {
    let m = model(ADDITIVE_MODEL);
    let pop = pop_with_crcl(&CRCL_GRID);
    let omega = diag_omega(&[0.2, 0.1]);
    let g = only_group(&m, &pop, &omega);
    let fixed = [false; 3];
    let etas = vec![vec![0.0, 0.0]; pop.subjects.len()];
    let input_for = |theta: &[f64; 3]| -> ([f64; 3], Vec<Vec<f64>>) { (*theta, etas.clone()) };

    // Admissible: TVCL + (CRCL−90)·TH_CRCL > 0 for CRCL = 40 (150 − 100 = 50).
    let ok = [150.0, 2.0, 10.0];
    // Inadmissible: at TH_CRCL = 4, CRCL = 40 gives 150 − 200 = −50, so `mu`
    // is NaN for that subject and the lognormal link has nothing to take a log
    // of. Every other subject is fine — one bad subject is enough.
    let bad = [150.0, 4.0, 10.0];
    let mus_bad = g.mus(&bad, &pop);
    assert!(
        mus_bad.iter().any(|m| !m.is_finite()),
        "premise: the fixture actually produces a non-finite mu, got {mus_bad:?}"
    );

    for (label, theta, want) in [("admissible", ok, true), ("inadmissible", bad, false)] {
        let (t, e) = input_for(&theta);
        let input = GroupStepInput {
            theta: &t,
            theta_lower: &m.default_params.theta_lower,
            theta_upper: &m.default_params.theta_upper,
            theta_fixed: &fixed,
            theta_packs_log: &[true, true, true],
            omega: &omega,
            etas: &e,
        };
        assert_eq!(
            g.can_step(&t, &pop, &fixed),
            want,
            "[{label}] can_step disagreed"
        );
        assert_eq!(
            g.solve_exact(&pop, &input).is_some(),
            want,
            "[{label}] solve_exact disagreed with can_step"
        );
        assert_eq!(
            g.solve_numerical(&pop, &input, 5, &|_t, _s| 0.0).is_some(),
            want,
            "[{label}] solve_numerical disagreed with can_step"
        );
    }

    // The other half of the predicate: an all-FIXed group has nothing to move,
    // whatever the typical value does.
    let all_fixed = [true; 3];
    assert!(!g.can_step(&ok, &pop, &all_fixed));
}

/// The group step returns the best point the objective was *evaluated* at, not
/// whatever `nlopt::optimize` happens to leave in its output buffer.
///
/// Honest scope: this pins a postcondition, it is not a regression test — the
/// reviewer's scenario (BOBYQA wandering a `1e20` sentinel plateau and handing
/// back a point worse than θ_old) could not be reproduced through this API,
/// because BOBYQA evaluates θ_old first and returns its own best. The tracking
/// makes the result independent of that library guarantee at zero extra
/// objective evaluations, and this test fails if a future edit returns the raw
/// `x` after post-processing it, or stops recording the start.
#[test]
fn numerical_solver_returns_the_best_evaluated_point() {
    let m = model(ADDITIVE_MODEL);
    let pop = pop_with_crcl(&CRCL_GRID);
    let omega = diag_omega(&[0.2, 0.1]);
    let g = only_group(&m, &pop, &omega);
    let start = [150.0, 2.0, 10.0];
    let etas = vec![vec![0.0, 0.0]; pop.subjects.len()];
    let fixed = [false; 3];
    let input = GroupStepInput {
        theta: &start,
        theta_lower: &m.default_params.theta_lower,
        theta_upper: &m.default_params.theta_upper,
        theta_fixed: &fixed,
        theta_packs_log: &[true, true, true],
        omega: &omega,
        etas: &etas,
    };
    // Record every (theta, data-term) the optimiser asked about. The prior term
    // is added by the solver on top, so the ranking below is reconstructed from
    // the *total*, recomputed here from the same pieces.
    let seen: std::cell::RefCell<Vec<Vec<f64>>> = std::cell::RefCell::new(Vec::new());
    let seen_ref = &seen;
    // A deceptive objective: lowest far from the start, so the answer is not
    // trivially θ_old, and rugged enough that BOBYQA's last trial point is not
    // its best one.
    let data = |theta: &[f64], _shift: &[f64]| -> f64 {
        seen_ref.borrow_mut().push(theta.to_vec());
        1e3 * ((theta[1] - 1.2) * (theta[1] - 1.2) + 0.001 * (theta[0] - 120.0).abs())
    };
    let out = g
        .solve_numerical(&pop, &input, 50, &data)
        .expect("solvable");
    let evaluated = seen.into_inner();
    assert!(
        evaluated.len() > 2,
        "premise: the objective ran more than once, got {}",
        evaluated.len()
    );
    // The returned point must be one of the evaluated ones …
    assert!(
        evaluated
            .iter()
            .any(|t| (t[0] - out[0]).abs() < 1e-9 && (t[1] - out[1]).abs() < 1e-9),
        "returned θ = ({}, {}) was never evaluated",
        out[0],
        out[1]
    );
    // … and none of them may beat it on the data term the test controls. (The
    // prior term is a function of the same θ through the shift, so a strict
    // total ordering on the data term alone is not guaranteed; what is checked
    // is that the return is not an *unevaluated* or a late-trial point, which
    // is how a raw-`x` return fails.)
    let out_data = 1e3 * ((out[1] - 1.2) * (out[1] - 1.2) + 0.001 * (out[0] - 120.0).abs());
    let best_data = evaluated
        .iter()
        .map(|t| 1e3 * ((t[1] - 1.2) * (t[1] - 1.2) + 0.001 * (t[0] - 120.0).abs()))
        .fold(f64::INFINITY, f64::min);
    assert!(
        out_data.is_finite() && best_data.is_finite(),
        "a non-finite fold would hide a diverged solve: out={out_data}, best={best_data}"
    );
    assert!(
        out_data <= best_data + 1e-6,
        "returned a point worse than one already evaluated: {out_data} vs {best_data}"
    );
}
