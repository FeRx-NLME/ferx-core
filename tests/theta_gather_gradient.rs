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
    fit_on(model_text, DATA)
}

fn fit_on(model_text: &str, data: &str) -> FitResult {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_path = dir.path().join("m.ferx");
    let data_path = dir.path().join("d.csv");
    write!(std::fs::File::create(&model_path).unwrap(), "{model_text}").unwrap();
    write!(std::fs::File::create(&data_path).unwrap(), "{data}").unwrap();
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
/// [`assert_analytic_ran_sees_a_runtime_decline`] is the in-file positive
/// control.
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
fn assert_analytic_ran_sees_a_runtime_decline() {
    // The case `assert_analytic_ran` exists for: the model-level label reads
    // "analytic" while the provider declines a subject at run time. A
    // rate-defined infusion under an `F` (the #419 decline, as in
    // `tests/outer_gradient_fd_fallback_warning.rs`) on subject 1 of the counted
    // form does that. A gather-free slot and the wide-block ladder do not: the
    // column-chunked jet keeps a 26-level block analytic, and
    // `reconverge_gradient_interval` forces FD without logging a decline.
    let form = &FORMS[0];
    let text = model(form, form.gather_line, SHORT_FIT)
        .replace("  theta TVV(", "  theta TVF(0.7, 0.05, 1.0)\n  theta TVV(")
        .replace("V = TVV * exp(ETA_V)", "V = TVV * exp(ETA_V)\n  F = TVF")
        .replace(
            "pk one_cpt_iv(cl=CL, v=V)",
            "pk one_cpt_iv(cl=CL, v=V, f=F)",
        );
    assert!(
        text.matches("TVF").count() == 2 && text.contains("f=F"),
        "every edit must take: {text}"
    );
    let data = DATA.replacen("1,0,.,1,100,1,0,1,1\n", "1,0,.,1,100,1,50,1,1\n", 1);
    assert_ne!(data, DATA, "subject 1's dose must become an infusion");
    let result = fit_on(&text, &data);
    assert!(
        result.gradient_method_outer.starts_with("analytic"),
        "the label must stay analytic for this to be the case under test: {}",
        result.gradient_method_outer
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains(OUTER_FD_FALLBACK)),
        "the declined subject must carry the fallback warning: {:?}",
        result.warnings
    );
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_analytic_ran(&result, "runtime-decline control")
    }));
    assert!(
        caught.is_err(),
        "assert_analytic_ran must reject a fit whose subject fell back to FD"
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

/// The exit criterion of #1628: on the default gradient the block lands at the
/// finite-difference optimum. A fit to convergence, hence gated.
///
/// **FD is the less accurate party, and how much less depends on the platform.**
/// At `f94eebed`/`4f7e7d0f` the analytic OFV is −134.0897788769 in every form,
/// on macOS and on the Linux CI runner alike, to 10 digits. FD stops *above* it:
///
/// | | counted, `contrast = none` | sum-to-zero |
/// |---|---|---|
/// | macOS, OFV analytic − FD | −6.8e-5 | −2.5e-4 |
/// | Linux, OFV analytic − FD | −4.12e-3 | not yet measured |
/// | macOS, worst CL gap | 7.6e-5 | 2.4e-4 |
/// | Linux, worst CL gap | 8.47e-4 (study 1) | not yet measured |
///
/// So the two checks against FD are shaped by that:
///
/// - **OFV, one-sided:** the analytic optimum may not sit above FD's by more
///   than `OFV_SLACK`. A two-sided bound measured FD's stopping point, not the
///   fix: the first one, at 3e-3, went red on Linux (slow-tests run
///   37112687521) with the analytic side right. Measured with the fix reverted
///   (`PushThetaGather` back on the `false` arm, macOS): the analytic OFV is
///   +78.68, **212.8 above** FD's in every form, so the one-sided bound still
///   kills that regression by five orders of magnitude.
/// - **CL, two-sided, at FD's accuracy:** 1e-2, 12× the Linux counted gap.
///   With the fix reverted every CL sits at its init, 1.5 (or `TVCL · e⁰`
///   = 1.56 for sum-to-zero), 21–59 % from FD's.
///
/// The FD-free oracle is the tight one: the three forms are reparameterisations
/// of one model, so their analytic optima imply the same CLs. Measured 7.0e-8
/// (`contrast = none` to the bit), bounded at 1e-6.
///
/// Every number is printed for every form before anything is asserted, so a
/// red run on one platform still reports the other forms.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_gathered_theta_converges_to_the_finite_difference_optimum() {
    const CONVERGE: &str = "  maxiter = 500";
    const OFV_SLACK: f64 = 1e-3;
    const CL_REL_TOL: f64 = 1e-2;
    const CROSS_FORM_TOL: f64 = 1e-6;
    let mut analytic_cl: Vec<[f64; 3]> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
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
        let (cl_a, cl_f) = (study_cl(form, &analytic), study_cl(form, &fd));
        assert!(
            analytic.ofv.is_finite()
                && fd.ofv.is_finite()
                && cl_a.iter().chain(&cl_f).all(|c| c.is_finite()),
            "{}: OFV {} / {}, CL {cl_a:?} / {cl_f:?}",
            form.name,
            analytic.ofv,
            fd.ofv
        );
        let d_ofv = analytic.ofv - fd.ofv;
        let cl_rel: Vec<f64> = (0..3)
            .map(|k| (cl_a[k] - cl_f[k]).abs() / cl_f[k])
            .collect();
        let cl_rel_txt: Vec<String> = cl_rel.iter().map(|r| format!("{r:.3e}")).collect();
        eprintln!(
            "{}: OFV analytic {:.10} fd {:.10} (analytic − fd {d_ofv:.3e}); \
             CL analytic {cl_a:?} fd {cl_f:?} (rel {cl_rel_txt:?})",
            form.name, analytic.ofv, fd.ofv
        );
        if d_ofv >= OFV_SLACK {
            failures.push(format!(
                "{}: the analytic OFV {} sits {d_ofv:.3e} above FD's {}",
                form.name, analytic.ofv, fd.ofv
            ));
        }
        for (k, rel) in cl_rel.iter().enumerate() {
            if *rel >= CL_REL_TOL {
                failures.push(format!(
                    "{}: study {} CL analytic {} vs fd {} (rel {rel:.3e})",
                    form.name,
                    k + 1,
                    cl_a[k],
                    cl_f[k]
                ));
            }
        }
        analytic_cl.push(cl_a);
    }
    for (form, cl) in FORMS.iter().zip(&analytic_cl).skip(1) {
        for k in 0..3 {
            let rel = (cl[k] - analytic_cl[0][k]).abs() / analytic_cl[0][k];
            eprintln!("{} vs counted, study {}: rel {rel:.3e}", form.name, k + 1);
            if rel >= CROSS_FORM_TOL {
                failures.push(format!(
                    "{}: study {} CL {} vs counted {} (rel {rel:.3e})",
                    form.name,
                    k + 1,
                    cl[k],
                    analytic_cl[0][k]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ── #1636: the same block read in a Form-C readout ──────────────────────────
//
// `y = central / V * SCALE[STUDY]` (ODE or closed form) and a compartment-free
// `y = PLACEBO[STUDY] + …` read the block in the readout, not in
// `[individual_parameters]`. The readout's `Dual2` walk had no reader for the
// gather, so every subject's analytic gradient came back NaN and fell back to FD
// while `optimizer = auto` stayed on L-BFGS: B1 (ODE) stopped at OFV −56.99 and
// F1 (closed form) at −59.07, against −108.28 for the same model with the gather
// written in `[individual_parameters]` (#1636 plan §0). The parser now lifts the
// readout gather into a synthetic individual parameter, so the two spellings run
// one individual-parameter program; the hand-lifted spelling is the oracle.

/// Three studies × four subjects × six times, compartment-free, drawn once from
/// `y = PL[STUDY] + EMAX·t/(t+2)` with `PL = (1, 2, 3)`, `EMAX = 4·exp(η)`,
/// ω = 0.04 and an additive residual SD of 0.3 (Python `random.seed(1636)`).
const CF_DATA: &str = "\
ID,TIME,DV,MDV,STUDY,ARM,NARM
1,0,0.7696,0,1,1,20
1,1,3.0163,0,1,1,20
1,2,3.0393,0,1,1,20
1,4,4.2464,0,1,1,20
1,8,4.8811,0,1,1,20
1,12,4.5333,0,1,1,20
2,0,1.4394,0,1,2,30
2,1,2.2587,0,1,2,30
2,2,2.9528,0,1,2,30
2,4,3.1951,0,1,2,30
2,8,3.9133,0,1,2,30
2,12,4.1039,0,1,2,30
3,0,0.8225,0,1,1,40
3,1,2.6367,0,1,1,40
3,2,3.8186,0,1,1,40
3,4,3.4628,0,1,1,40
3,8,4.2179,0,1,1,40
3,12,5.2294,0,1,1,40
4,0,0.8663,0,1,2,50
4,1,2.2287,0,1,2,50
4,2,3.0410,0,1,2,50
4,4,3.5308,0,1,2,50
4,8,3.5488,0,1,2,50
4,12,4.4198,0,1,2,50
5,0,1.5308,0,2,1,20
5,1,4.1405,0,2,1,20
5,2,5.1515,0,2,1,20
5,4,5.7832,0,2,1,20
5,8,6.3899,0,2,1,20
5,12,7.2902,0,2,1,20
6,0,1.6006,0,2,2,30
6,1,3.1121,0,2,2,30
6,2,3.4515,0,2,2,30
6,4,4.3426,0,2,2,30
6,8,4.8627,0,2,2,30
6,12,4.9546,0,2,2,30
7,0,1.8507,0,2,1,40
7,1,3.0570,0,2,1,40
7,2,3.3000,0,2,1,40
7,4,4.5229,0,2,1,40
7,8,4.5731,0,2,1,40
7,12,4.8838,0,2,1,40
8,0,2.3240,0,2,2,50
8,1,3.9706,0,2,2,50
8,2,5.0714,0,2,2,50
8,4,5.5275,0,2,2,50
8,8,6.5195,0,2,2,50
8,12,6.3334,0,2,2,50
9,0,2.6015,0,3,1,20
9,1,4.4499,0,3,1,20
9,2,5.1670,0,3,1,20
9,4,6.0853,0,3,1,20
9,8,6.4819,0,3,1,20
9,12,7.0542,0,3,1,20
10,0,2.9125,0,3,2,30
10,1,4.3693,0,3,2,30
10,2,5.0397,0,3,2,30
10,4,5.9698,0,3,2,30
10,8,5.9623,0,3,2,30
10,12,6.6390,0,3,2,30
11,0,3.3501,0,3,1,40
11,1,3.6714,0,3,1,40
11,2,4.8126,0,3,1,40
11,4,4.7174,0,3,1,40
11,8,4.7759,0,3,1,40
11,12,5.6959,0,3,1,40
12,0,2.8934,0,3,2,50
12,1,4.9183,0,3,2,50
12,2,5.4106,0,3,2,50
12,4,6.3672,0,3,2,50
12,8,7.2298,0,3,2,50
12,12,7.2235,0,3,2,50
";

/// One engine's readout fixture: the model with the gather in `y`, and its
/// hand-lifted twin with the gather in `[individual_parameters]`.
struct Readout {
    engine: &'static str,
    data: &'static str,
    in_y: fn(&str, &str) -> String,
    lifted: fn(&str, &str) -> String,
}

fn pk_readout_model(structural: &str, theta: &str, ip: &str, y: &str, fo: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.01, 20.0)
  theta TVV(8.0, 0.1, 500.0)
  {theta}
  omega ETA_V ~ 0.04
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL
  V = TVV * exp(ETA_V)
  {ip}

[structural_model]
{structural}

[scaling]
  y = {y}

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
  covariance = false
{fo}
"#
    )
}

const ODE_STRUCTURAL: &str =
    "  ode(states=[central])\n\n[odes]\n  d/dt(central) = -CL / V * central";
const CLOSED_STRUCTURAL: &str = "  pk one_cpt_iv(cl=CL, v=V)";

fn cf_readout_model(theta: &str, ip: &str, y: &str, fo: &str) -> String {
    format!(
        r#"
[parameters]
  {theta}
  theta TVEMAX(3.0, 0.1, 20.0)
  theta TVET50(1.5, 0.1, 20.0)
  omega ETA_EMAX ~ 0.04
  sigma ADD ~ 0.1

[individual_parameters]
  EMAX = TVEMAX * exp(ETA_EMAX)
  ET50 = TVET50
  {ip}

[structural_model]
  y = {y}

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = focei
  covariance = false
{fo}
"#
    )
}

const CF_EFFECT: &str = "EMAX * TIME / (TIME + ET50)";

/// The three engines behind `eval_output_g`, each with the counted block
/// indexed by `STUDY`. `theta` is the block's declaration line.
const READOUTS: [Readout; 3] = [
    Readout {
        engine: "ODE Form-C (sens/ode_provider.rs)",
        data: DATA,
        in_y: |theta, fo| {
            pk_readout_model(ODE_STRUCTURAL, theta, "", "central / V * SCALE[STUDY]", fo)
        },
        lifted: |theta, fo| {
            pk_readout_model(
                ODE_STRUCTURAL,
                theta,
                "S = SCALE[STUDY]",
                "central / V * S",
                fo,
            )
        },
    },
    Readout {
        engine: "closed-form Form-C (sens/provider.rs)",
        data: DATA,
        in_y: |theta, fo| {
            pk_readout_model(
                CLOSED_STRUCTURAL,
                theta,
                "",
                "central / V * SCALE[STUDY]",
                fo,
            )
        },
        lifted: |theta, fo| {
            pk_readout_model(
                CLOSED_STRUCTURAL,
                theta,
                "S = SCALE[STUDY]",
                "central / V * S",
                fo,
            )
        },
    },
    Readout {
        engine: "compartment-free (sens/algebraic.rs)",
        data: CF_DATA,
        in_y: |theta, fo| {
            cf_readout_model(
                &theta.replace("SCALE", "PLACEBO"),
                "",
                &format!("PLACEBO[STUDY] + {CF_EFFECT}"),
                fo,
            )
        },
        lifted: |theta, fo| {
            cf_readout_model(
                &theta.replace("SCALE", "PLACEBO"),
                "E0 = PLACEBO[STUDY]",
                &format!("E0 + {CF_EFFECT}"),
                fo,
            )
        },
    },
];

const READOUT_BLOCK: &str = "theta SCALE[3](1.5, 0.1, 10.0)";
const READOUT_INIT: f64 = 1.5;

/// Every estimated level of the readout's block (`SCALE` or `PLACEBO`).
fn readout_levels(result: &FitResult) -> Vec<(String, f64)> {
    let out: Vec<(String, f64)> = result
        .theta_names
        .iter()
        .zip(&result.theta)
        .filter(|(n, _)| n.starts_with("SCALE[") || n.starts_with("PLACEBO["))
        .map(|(n, v)| (n.clone(), *v))
        .collect();
    assert_eq!(out.len(), 3, "three levels in {:?}", result.theta_names);
    out
}

/// T4 + T6, per engine: the gather in `y` runs on the analytic gradient with no
/// subject falling back, moves the block, names no level "not referenced", and
/// fits exactly like the hand-lifted twin. The pair straddles the fix: before it,
/// the in-`y` side fell back on 12/12 subjects (and `assert_analytic_ran` fails)
/// while the lifted side did not.
#[test]
fn a_readout_gather_fits_exactly_like_its_lifted_twin() {
    for r in &READOUTS {
        let in_y = fit_on(&(r.in_y)(READOUT_BLOCK, SHORT_FIT), r.data);
        let lifted = fit_on(&(r.lifted)(READOUT_BLOCK, SHORT_FIT), r.data);
        assert_analytic_ran(&in_y, r.engine);
        assert_analytic_ran(&lifted, r.engine);
        let unreferenced: Vec<&String> = in_y
            .warnings
            .iter()
            .filter(|w| w.contains("but not referenced"))
            .collect();
        assert!(
            unreferenced.is_empty(),
            "{}: a level read in the readout reported unreferenced: {unreferenced:?}",
            r.engine
        );
        for (name, v) in readout_levels(&in_y) {
            assert!(
                v.is_finite() && (v - READOUT_INIT).abs() > 1e-3,
                "{}: {name} = {v} did not leave its init",
                r.engine
            );
        }
        assert!(in_y.ofv.is_finite(), "{}: OFV {}", r.engine, in_y.ofv);
        assert_eq!(
            in_y.ofv.to_bits(),
            lifted.ofv.to_bits(),
            "{}: OFV {:.17e} vs lifted {:.17e}",
            r.engine,
            in_y.ofv,
            lifted.ofv
        );
        assert_eq!(in_y.theta_names, lifted.theta_names, "{}", r.engine);
        for (a, b) in in_y.theta.iter().zip(&lifted.theta) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{}: θ {:?} vs lifted {:?}",
                r.engine,
                in_y.theta,
                lifted.theta
            );
        }
    }
}

/// Population predictions of `model` on `data` at `theta_of(name, default)`,
/// as `(id, time, pred)` per observation.
fn level_preds(model: &str, data: &str) -> Vec<(String, f64, f64)> {
    let dir = tempfile::tempdir().expect("tempdir");
    let data_path = dir.path().join("d.csv");
    write!(std::fs::File::create(&data_path).unwrap(), "{data}").unwrap();
    let mut parsed = ferx_core::parser::model_parser::parse_full_model(model).expect("parse");
    let mut pop = ferx_core::read_nonmem_csv(&data_path, None, None).expect("data");
    ferx_core::bind_theta_levels(&mut parsed, model, &mut pop).expect("bind");
    let mut params = parsed.model.default_params.clone();
    // A distinct value per level, so a synth that read the wrong level (or one
    // level for every row) cannot agree with the twin by coincidence.
    for (i, n) in parsed.model.theta_names.iter().enumerate() {
        if let Some(at) = n.find('[') {
            let h: u32 = n[at..].bytes().map(u32::from).sum();
            params.theta[i] = 1.0 + f64::from(h % 17) / 10.0;
        }
    }
    let rows = ferx_core::predict(&parsed.model, &pop, &params).expect("predict");
    rows.iter()
        .map(|r| (r.id.clone(), r.time, r.pred))
        .collect()
}

/// T5: `predict()` is bit-identical between the gather in `y` and the gather
/// lifted by hand, on every engine and on an index that varies per observation
/// (`[STUDY, TIME]`), so the synth is evaluated with each observation's
/// covariates, not the subject's first row.
#[test]
fn a_readout_gather_predicts_exactly_like_its_lifted_twin() {
    let named = [
        "theta SCALE[STUDY, contrast = none](1.0, 0.1, 10.0)",
        "theta SCALE[STUDY, TIME, contrast = none](1.0, 0.1, 10.0)",
        "theta SCALE[STUDY, TIME, contrast = sum_to_zero](0.0, -10.0, 10.0)",
    ];
    let mut cases: Vec<(String, String, String, &str)> = Vec::new();
    for theta in named {
        for (engine, structural) in [("ODE", ODE_STRUCTURAL), ("closed form", CLOSED_STRUCTURAL)] {
            cases.push((
                format!("{engine}, {theta}"),
                pk_readout_model(structural, theta, "", "central / V * exp(SCALE)", ""),
                pk_readout_model(structural, theta, "S = SCALE", "central / V * exp(S)", ""),
                DATA,
            ));
        }
        let theta = theta.replace("SCALE", "PLACEBO");
        cases.push((
            format!("compartment-free, {theta}"),
            cf_readout_model(&theta, "", &format!("PLACEBO + {CF_EFFECT}"), ""),
            cf_readout_model(&theta, "E0 = PLACEBO", &format!("E0 + {CF_EFFECT}"), ""),
            CF_DATA,
        ));
    }
    // An index read through an individual parameter.
    cases.push((
        "ODE, SCALE[K] with K = STUDY".into(),
        pk_readout_model(
            ODE_STRUCTURAL,
            "theta SCALE[3](1.0, 0.1, 10.0)",
            "K = STUDY",
            "central / V * SCALE[K]",
            "",
        ),
        pk_readout_model(
            ODE_STRUCTURAL,
            "theta SCALE[3](1.0, 0.1, 10.0)",
            "K = STUDY\n  S = SCALE[K]",
            "central / V * S",
            "",
        ),
        DATA,
    ));
    for (tag, in_y, lifted, data) in &cases {
        let (a, b) = (level_preds(in_y, data), level_preds(lifted, data));
        assert_eq!(a.len(), b.len(), "{tag}");
        assert!(a.len() >= 60, "{tag}: {} rows", a.len());
        let distinct: std::collections::BTreeSet<u64> = a.iter().map(|r| r.2.to_bits()).collect();
        assert!(distinct.len() > 3, "{tag}: the predictions must vary");
        for (x, y) in a.iter().zip(&b) {
            assert!(x.2.is_finite() && y.2.is_finite(), "{tag}: {x:?} / {y:?}");
            assert_eq!((&x.0, x.1), (&y.0, y.1), "{tag}: row order");
            assert_eq!(x.2.to_bits(), y.2.to_bits(), "{tag}: {x:?} vs lifted {y:?}");
        }
    }
}

/// The exit criterion of #1636: on the default gradient a readout gather reaches
/// the optimum of its hand-lifted twin, where before it stopped 49–51 OFV above
/// it on L-BFGS over FD gradients. A fit to convergence, hence gated.
///
/// **The primary oracle is FD-free:** the in-`y` and the hand-lifted spellings run
/// one individual-parameter program, so their converged OFVs must agree to the bit.
/// Measured at `4643eef7` + this change (macOS), bit-identical on all three engines:
/// ODE −108.2821614347, closed form −108.2777960837, compartment-free −78.0925575386.
///
/// **FD is secondary and one-sided** (the #1628 lesson: FD's stopping point moves by
/// platform). FD stops *above* the analytic optimum on every engine (analytic − FD:
/// ODE −7.4e-1, closed form −8.1e-3, compartment-free −5.6e-3), so the analytic OFV
/// may not sit above FD's by more than `FD_OFV_SLACK = 1e-2`. Before the fix the
/// in-`y` side sat +47.8 (ODE) and +49.2 (closed form) above FD, and +3.4e-2 on
/// the compartment-free engine (#1636 plan §0, B1/F1/A1), so the slack still kills
/// all three.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_readout_gather_converges_to_the_lifted_optimum() {
    const CONVERGE: &str = "  maxiter = 500";
    const FD_OFV_SLACK: f64 = 1e-2;
    let mut failures: Vec<String> = Vec::new();
    for r in &READOUTS {
        let in_y = fit_on(&(r.in_y)(READOUT_BLOCK, CONVERGE), r.data);
        let lifted = fit_on(&(r.lifted)(READOUT_BLOCK, CONVERGE), r.data);
        let fd = fit_on(
            &(r.in_y)(READOUT_BLOCK, &format!("{CONVERGE}\n  gradient = fd")),
            r.data,
        );
        assert_analytic_ran(&in_y, r.engine);
        assert!(
            in_y.ofv.is_finite() && lifted.ofv.is_finite() && fd.ofv.is_finite(),
            "{}: OFV {} / {} / {}",
            r.engine,
            in_y.ofv,
            lifted.ofv,
            fd.ofv
        );
        let d_fd = in_y.ofv - fd.ofv;
        eprintln!(
            "{}: OFV in y {:.10}, lifted {:.10}, fd {:.10} (in y − fd {d_fd:.3e}); levels {:?}",
            r.engine,
            in_y.ofv,
            lifted.ofv,
            fd.ofv,
            readout_levels(&in_y)
        );
        if in_y.ofv.to_bits() != lifted.ofv.to_bits() {
            failures.push(format!(
                "{}: OFV {} vs lifted {}",
                r.engine, in_y.ofv, lifted.ofv
            ));
        }
        if d_fd >= FD_OFV_SLACK {
            failures.push(format!(
                "{}: the analytic OFV {} sits {d_fd:.3e} above FD's {}",
                r.engine, in_y.ofv, fd.ofv
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
