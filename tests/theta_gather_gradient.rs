//! A θ read through a gather is estimated on the default (analytic) gradient (#1628).
//!
//! The individual-parameter dual walk folds a slot that reads no θ/η to a
//! constant, computed once in `f64` (#485). Its bytecode classifier did not count
//! `PushThetaGather` as a θ read, so a slot whose *only* θ read is a gather —
//! `CL = PLACEBO[STUDY]`, a level block's bare `PLACEBO`, or an intermediate
//! `PL = PLACEBO` — got a zero analytic gradient in every gathered θ, and the
//! fit returned the whole block at its initial values. `gradient = fd` moved it;
//! so did a literal `+ 0 * <any θ>` on the same line, which made the slot
//! dynamic. The parse-time "declared but not referenced" check had the same
//! blind spot and flagged every level.
//!
//! Each case runs in three forms: the counted block indexed by a data column,
//! the named-level block with no contrast (the same levels, bit for bit), and the
//! named-level block under global sum-to-zero through an intermediate, which
//! puts a `NegSum` level behind the gather.

use ferx_core::{run_model_with_data, FitResult};
use std::io::Write;

/// Three studies × four subjects × five samples, 1-cpt IV, drawn once from
/// CL = 1.0 / 2.0 / 3.5 by study, V = 10·exp(η), η ~ N(0, 0.04), 10%
/// proportional error (Python `random.seed(1628)`). Every level has twenty
/// observations, so unlike a one-sample-per-level design the gathered θ are
/// identified and two optimizers have one optimum to agree on.
const DATA: &str = "\
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,STUDY
1,0,.,1,100,1,0,1,1
1,0.5,7.9799,0,.,1,0,0,1
1,2,6.4449,0,.,1,0,0,1
1,6,5.3060,0,.,1,0,0,1
1,12,3.0189,0,.,1,0,0,1
1,24,1.4837,0,.,1,0,0,1
2,0,.,1,100,1,0,1,1
2,0.5,8.2219,0,.,1,0,0,1
2,2,7.0981,0,.,1,0,0,1
2,6,5.5380,0,.,1,0,0,1
2,12,2.5755,0,.,1,0,0,1
2,24,0.9625,0,.,1,0,0,1
3,0,.,1,100,1,0,1,1
3,0.5,15.2808,0,.,1,0,0,1
3,2,9.9433,0,.,1,0,0,1
3,6,6.0395,0,.,1,0,0,1
3,12,2.8442,0,.,1,0,0,1
3,24,0.4437,0,.,1,0,0,1
4,0,.,1,100,1,0,1,1
4,0.5,10.2019,0,.,1,0,0,1
4,2,8.3891,0,.,1,0,0,1
4,6,5.0451,0,.,1,0,0,1
4,12,3.0938,0,.,1,0,0,1
4,24,0.9359,0,.,1,0,0,1
5,0,.,1,100,1,0,1,2
5,0.5,10.0458,0,.,1,0,0,2
5,2,5.5739,0,.,1,0,0,2
5,6,2.7556,0,.,1,0,0,2
5,12,1.0420,0,.,1,0,0,2
5,24,0.0950,0,.,1,0,0,2
6,0,.,1,100,1,0,1,2
6,0.5,8.2184,0,.,1,0,0,2
6,2,7.3120,0,.,1,0,0,2
6,6,3.3537,0,.,1,0,0,2
6,12,0.9309,0,.,1,0,0,2
6,24,0.0707,0,.,1,0,0,2
7,0,.,1,100,1,0,1,2
7,0.5,9.5787,0,.,1,0,0,2
7,2,8.5524,0,.,1,0,0,2
7,6,2.6363,0,.,1,0,0,2
7,12,0.7048,0,.,1,0,0,2
7,24,0.0349,0,.,1,0,0,2
8,0,.,1,100,1,0,1,2
8,0.5,7.5717,0,.,1,0,0,2
8,2,6.3016,0,.,1,0,0,2
8,6,3.1313,0,.,1,0,0,2
8,12,1.0668,0,.,1,0,0,2
8,24,0.1804,0,.,1,0,0,2
9,0,.,1,100,1,0,1,3
9,0.5,8.4379,0,.,1,0,0,3
9,2,4.7808,0,.,1,0,0,3
9,6,1.1073,0,.,1,0,0,3
9,12,0.1123,0,.,1,0,0,3
9,24,0.0017,0,.,1,0,0,3
10,0,.,1,100,1,0,1,3
10,0.5,8.5898,0,.,1,0,0,3
10,2,4.4894,0,.,1,0,0,3
10,6,1.1527,0,.,1,0,0,3
10,12,0.1491,0,.,1,0,0,3
10,24,0.0022,0,.,1,0,0,3
11,0,.,1,100,1,0,1,3
11,0.5,6.4659,0,.,1,0,0,3
11,2,4.7337,0,.,1,0,0,3
11,6,1.5048,0,.,1,0,0,3
11,12,0.2669,0,.,1,0,0,3
11,24,0.0088,0,.,1,0,0,3
12,0,.,1,100,1,0,1,3
12,0.5,7.0205,0,.,1,0,0,3
12,2,5.5900,0,.,1,0,0,3
12,6,1.1611,0,.,1,0,0,3
12,12,0.1352,0,.,1,0,0,3
12,24,0.0020,0,.,1,0,0,3
";

/// One way of writing the study effect. `gather_line` is the single line that
/// reads the block; [`twin`] appends `+ 0 * TVV` to exactly that line.
struct Form {
    name: &'static str,
    theta_line: &'static str,
    gather_line: &'static str,
    cl_line: &'static str,
    /// The block's initial value, which an unestimated level keeps.
    init: f64,
}

const FORMS: [Form; 3] = [
    Form {
        name: "counted",
        theta_line: "theta PLACEBO[3](1.5, 0.01, 20.0)",
        gather_line: "CL = PLACEBO[STUDY]",
        cl_line: "",
        init: 1.5,
    },
    Form {
        name: "named, contrast = none",
        theta_line: "theta PLACEBO[STUDY, contrast = none](1.5, 0.01, 20.0)",
        gather_line: "CL = PLACEBO",
        cl_line: "",
        init: 1.5,
    },
    Form {
        name: "named, sum-to-zero",
        theta_line: "theta TVCL(2.0, 0.01, 20.0)\n  theta PLACEBO[STUDY](0.0, -5.0, 5.0)",
        gather_line: "PL = PLACEBO",
        cl_line: "CL = TVCL * exp(PL)",
        init: 0.0,
    },
];

fn model(form: &Form, gather_line: &str, fit_options: &str) -> String {
    format!(
        r#"
[parameters]
  {theta_line}
  theta TVV(8.0, 0.1, 500.0)
  omega ETA_V ~ 0.04
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  {gather_line}
  {cl_line}
  V = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
  covariance = false
{fit_options}
"#,
        theta_line = form.theta_line,
        cl_line = form.cl_line,
    )
}

/// The gather line with a θ read that changes nothing: `x + 0·TVV` is `x`
/// exactly, in value and in every dual component. It makes the slot dynamic
/// whatever the classifier thinks of the gather, so before the fix it was the
/// workaround (it moved the block) and after it the fit must equal it to the bit.
fn twin(form: &Form) -> String {
    format!("{} + 0 * TVV", form.gather_line)
}

fn fit(model_text: &str) -> FitResult {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_path = dir.path().join("m.ferx");
    let data_path = dir.path().join("d.csv");
    write!(std::fs::File::create(&model_path).unwrap(), "{model_text}").unwrap();
    write!(std::fs::File::create(&data_path).unwrap(), "{DATA}").unwrap();
    let (result, _pop) = run_model_with_data(
        model_path.to_str().unwrap(),
        Some(data_path.to_str().unwrap()),
    )
    .expect("fit");
    result
}

/// `(θ index, value)` of every estimated level of the block.
fn levels(result: &FitResult) -> Vec<(usize, f64)> {
    let out: Vec<(usize, f64)> = result
        .theta_names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.starts_with("PLACEBO["))
        .map(|(i, _)| (i, result.theta[i]))
        .collect();
    assert!(
        !out.is_empty(),
        "no PLACEBO level in {:?}",
        result.theta_names
    );
    out
}

const SHORT_FIT: &str = "  maxiter = 3\n  inner_maxiter = 20";

/// The token of `outer_fd_fallback_warning`: some subjects' outer gradient was
/// not the exact analytic one at run time. The same phrase that
/// `tests/outer_gradient_fd_fallback_warning.rs` matches, where
/// `out_of_scope_subject_warns_while_the_report_still_says_analytic` produces
/// the warning in a real fit, so a wording change reddens that test rather than
/// silently emptying the check below.
///
/// No positive control here: on this model the provider declines nothing. A
/// block past the 24-axis dual ladder is not a trigger either (measured: 26
/// levels, one read per subject, ran analytic through the column-chunked jet),
/// and `reconverge_gradient_interval` forces FD without logging a decline.
const OUTER_FD_FALLBACK: &str = "could not be given the exact analytic outer gradient";

/// The analytic outer gradient actually ran on every subject.
///
/// `gradient_method_outer` alone cannot say so: it is the *model-level* route
/// (`build_info::gradient_method_outer` says it "must not be used as a gate for
/// 'did the analytic outer gradient run'"), and a subject moved onto FD at run
/// time leaves it reading "analytic". The runtime truth is the fallback warning,
/// so require the label *and* its absence.
fn assert_analytic_ran(result: &FitResult, case: &str) {
    assert!(
        result.gradient_method_outer.starts_with("analytic"),
        "{case}: the model-level route must be the analytic gradient under test, got {}",
        result.gradient_method_outer
    );
    let fallback: Vec<&String> = result
        .warnings
        .iter()
        .filter(|w| w.contains(OUTER_FD_FALLBACK))
        .collect();
    assert!(
        fallback.is_empty(),
        "{case}: subjects fell back off the analytic outer gradient: {fallback:?}"
    );
}

#[test]
fn a_gathered_theta_leaves_its_init_on_the_analytic_gradient() {
    for form in &FORMS {
        let result = fit(&model(form, form.gather_line, SHORT_FIT));
        assert_analytic_ran(&result, form.name);
        for (i, v) in levels(&result) {
            assert!(
                v.is_finite(),
                "{}: {} = {v}",
                form.name,
                result.theta_names[i]
            );
            assert!(
                (v - form.init).abs() > 1e-3,
                "{}: {} stayed at its init {} — the analytic gradient does not \
                 reach a θ read through a gather",
                form.name,
                result.theta_names[i],
                form.init
            );
        }
    }
}

#[test]
fn a_gathered_theta_fits_exactly_like_its_directly_read_twin() {
    // The differential pair straddles the fix: before it, the gather-only slot
    // folded (block frozen) and the twin's did not (block moved), so the two
    // disagreed. Both sides must therefore move the block, or the equality below
    // could hold because neither does.
    for form in &FORMS {
        let gathered = fit(&model(form, form.gather_line, SHORT_FIT));
        let direct = fit(&model(form, &twin(form), SHORT_FIT));
        assert_eq!(gathered.theta_names, direct.theta_names, "{}", form.name);
        for (i, v) in levels(&direct) {
            assert!(
                (v - form.init).abs() > 1e-3,
                "{}: the twin must move {} for the pair to straddle",
                form.name,
                direct.theta_names[i]
            );
        }
        assert!(
            gathered.ofv.is_finite(),
            "{}: OFV {}",
            form.name,
            gathered.ofv
        );
        assert_eq!(
            gathered.ofv.to_bits(),
            direct.ofv.to_bits(),
            "{}: OFV {:.17e} vs twin {:.17e}",
            form.name,
            gathered.ofv,
            direct.ofv
        );
        for (a, b) in gathered.theta.iter().zip(&direct.theta) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{}: θ {:?} vs twin {:?}",
                form.name,
                gathered.theta,
                direct.theta
            );
        }
    }
}

#[test]
fn a_gathered_theta_is_not_reported_as_unreferenced() {
    // The parse-time check flagged every level of a gathered block as "declared
    // … but not referenced". A θ that really is unread must still be flagged, so
    // the check is not simply off: `UNUSED` is that control.
    for form in &FORMS {
        let text = model(form, form.gather_line, SHORT_FIT).replace(
            "  theta TVV(",
            "  theta UNUSED(1.0, 0.1, 10.0)\n  theta TVV(",
        );
        let result = fit(&text);
        let unreferenced: Vec<&String> = result
            .warnings
            .iter()
            .filter(|w| w.contains("but not referenced"))
            .collect();
        assert!(
            unreferenced.iter().all(|w| !w.contains("PLACEBO")),
            "{}: a gathered level reported unreferenced: {unreferenced:?}",
            form.name
        );
        assert_eq!(
            unreferenced.len(),
            1,
            "{}: exactly the control θ: {unreferenced:?}",
            form.name
        );
        assert!(
            unreferenced[0].contains("'UNUSED'"),
            "{}: {unreferenced:?}",
            form.name
        );
    }
}

/// Per-study clearance a fit implies, `[CL₁, CL₂, CL₃]`. The counted and
/// `contrast = none` forms estimate it directly; the sum-to-zero form as
/// `TVCL · exp(PL)`, with the dependent study's level the negated sum of the
/// free ones. This is the quantity all three forms share, and the one the data
/// identify: a sum-to-zero level near zero has no useful relative error.
fn study_cl(form: &Form, result: &FitResult) -> [f64; 3] {
    let by_study = |study: usize| {
        let name = match form.name {
            "counted" => format!("PLACEBO[{study}]"),
            _ => format!("PLACEBO[STUDY={study}]"),
        };
        result
            .theta_names
            .iter()
            .position(|n| *n == name)
            .map(|i| result.theta[i])
    };
    if form.init != 0.0 {
        return [1, 2, 3].map(|s| by_study(s).expect("every level is free"));
    }
    let tvcl_at = result.theta_names.iter().position(|n| n == "TVCL");
    let tvcl = result.theta[tvcl_at.expect("TVCL")];
    let free: Vec<Option<f64>> = (1..=3).map(by_study).collect();
    assert_eq!(
        free.iter().filter(|l| l.is_none()).count(),
        1,
        "one dependent level: {:?}",
        result.theta_names
    );
    let dependent = -free.iter().flatten().sum::<f64>();
    [0, 1, 2].map(|k| tvcl * free[k].unwrap_or(dependent).exp())
}

/// The exit criterion of #1628: on the default gradient the block lands where
/// finite differences put it. A fit to convergence, hence gated.
///
/// Measured on this fixture (`maxiter = 500`, macOS): the analytic OFV is
/// −134.0897788769 in every form, and FD stops 6.8e-5 (counted, `contrast =
/// none`) and 2.5e-4 (sum-to-zero) *above* it — FD is the less converged party.
/// The worst per-study CL gap is 2.4e-4 relative (sum-to-zero, study 2). The
/// bounds below are 12× and 8× those. Across forms the analytic optima agree to
/// 7.0e-8 (`contrast = none` to the bit), bounded at 1e-6. Before the fix every
/// CL here was its init, 1.5 (or `TVCL · e⁰`), against an optimum of 0.99 /
/// 1.98 / 3.63.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_gathered_theta_converges_to_the_finite_difference_optimum() {
    const CONVERGE: &str = "  maxiter = 500";
    const OFV_TOL: f64 = 3e-3;
    const CL_REL_TOL: f64 = 2e-3;
    const CROSS_FORM_TOL: f64 = 1e-6;
    let mut analytic_cl: Vec<[f64; 3]> = Vec::new();
    for form in &FORMS {
        let analytic = fit(&model(form, form.gather_line, CONVERGE));
        let fd = fit(&model(
            form,
            form.gather_line,
            &format!("{CONVERGE}\n  gradient = fd"),
        ));
        // Without the runtime half of this check, a fit whose subjects all fell
        // back to FD would make this an FD-vs-FD comparison and pass.
        assert_analytic_ran(&analytic, form.name);
        assert!(
            !fd.gradient_method_outer.starts_with("analytic"),
            "{}: {}",
            form.name,
            fd.gradient_method_outer
        );
        assert!(
            analytic.ofv.is_finite() && fd.ofv.is_finite(),
            "{}",
            form.name
        );
        let (cl_a, cl_f) = (study_cl(form, &analytic), study_cl(form, &fd));
        eprintln!(
            "{}: OFV analytic {:.10} fd {:.10} (Δ {:.2e}); CL analytic {cl_a:?} fd {cl_f:?}",
            form.name,
            analytic.ofv,
            fd.ofv,
            analytic.ofv - fd.ofv
        );
        assert!(
            (analytic.ofv - fd.ofv).abs() < OFV_TOL,
            "{}: OFV analytic {} vs fd {}",
            form.name,
            analytic.ofv,
            fd.ofv
        );
        for k in 0..3 {
            assert!(cl_a[k].is_finite() && cl_f[k].is_finite(), "{}", form.name);
            let rel = (cl_a[k] - cl_f[k]).abs() / cl_f[k];
            assert!(
                rel < CL_REL_TOL,
                "{}: study {} CL analytic {} vs fd {} (rel {rel:.3e})",
                form.name,
                k + 1,
                cl_a[k],
                cl_f[k]
            );
        }
        analytic_cl.push(cl_a);
    }
    // A second oracle that does not lean on FD: the three forms are
    // reparameterisations of one model, so their optima imply the same CLs.
    for (form, cl) in FORMS.iter().zip(&analytic_cl).skip(1) {
        for k in 0..3 {
            let rel = (cl[k] - analytic_cl[0][k]).abs() / analytic_cl[0][k];
            assert!(
                rel < CROSS_FORM_TOL,
                "{}: study {} CL {} vs counted {}",
                form.name,
                k + 1,
                cl[k],
                analytic_cl[0][k]
            );
        }
    }
}
