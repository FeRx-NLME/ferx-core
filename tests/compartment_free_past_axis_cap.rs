//! #2030: a compartment-free (`$PRED`-equivalent, #811) model past 24 θ + η keeps the
//! analytic route, and nothing advises FIXing parameters to get it back.
//!
//! Before #2093 the compartment-free provider capped `n_theta + n_eta` at 24, FIX θ
//! included, and a parse-time warning past the cap said the model "estimates 25
//! parameters" and advised to "FIX the ones you are not estimating". The advice did not
//! work: a model whose extra θ were all FIX got the same warning and the same route — a
//! finite-difference inner gradient and, under `optimizer = auto`, a derivative-free
//! BOBYQA outer loop (`gradient_method_outer = N/A`). The adjoint (#2093) has no θ cap,
//! and the warning went with the provider; this pins both.
//!
//! The fixture is the issue's: a compartment-free Emax model (3 θ, 1 η) padded with `n`
//! θ that enter `E0` as `+ 0.0 * PADi`, so the likelihood is identical at every pad count.
//! Three twins: 20 estimated pads (24 axes, at the old cap), 21 estimated pads and 21 FIX
//! pads (25 axes, past it). The FIX twin is the one the old advice pointed to. Each is
//! evaluated at the start values (`outer_maxiter = 0`, no covariance step) and must agree
//! with the others subject by subject.
//!
//! No NONMEM anchor: no numeric object is new; the likelihood is the 24-axis model's
//! (`algebraic_pred_nonmem_anchor.rs` anchors the compartment-free route itself). What this
//! pins is the route and the absence of the advice. Mutation (measured, macOS arm64):
//! reinstating an `n_theta + n_eta <= 24` gate in `sens::ode_adjoint::recorder_in_scope`
//! moves subject 2's per-subject OFV by one ulp on each 25-axis twin (3804.19808244241 vs
//! 3804.1980824424095 — the FD inner stops at a different η̂), and with the numeric checks
//! removed the FIX twin's inner label reads `finite differences`. Each 25-axis twin dies on
//! its own; the 24-axis twin stays green.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::FitOptions;
use ferx_core::{fit, read_nonmem_csv, FitResult};

/// The issue's synthetic data: 8 subjects × 8 observations (OCC is unused).
const DATA: &str = "\
ID,TIME,DV,EVID,AMT,CMT,MDV,OCC
1,1,12.0849,0,.,1,0,1
1,2,10.2993,0,.,1,0,1
1,4,8.8606,0,.,1,0,1
1,8,8.3164,0,.,1,0,1
1,11,9.8192,0,.,1,0,2
1,12,10.6304,0,.,1,0,2
1,14,10.5078,0,.,1,0,2
1,18,10.5978,0,.,1,0,2
2,1,5.5817,0,.,1,0,1
2,2,4.2526,0,.,1,0,1
2,4,3.6946,0,.,1,0,1
2,8,2.8419,0,.,1,0,1
2,11,-0.5909,0,.,1,0,2
2,12,-0.6863,0,.,1,0,2
2,14,-0.9173,0,.,1,0,2
2,18,-1.1169,0,.,1,0,2
3,1,11.3992,0,.,1,0,1
3,2,10.8766,0,.,1,0,1
3,4,9.7450,0,.,1,0,1
3,8,8.7293,0,.,1,0,1
3,11,9.2478,0,.,1,0,2
3,12,9.0279,0,.,1,0,2
3,14,9.2279,0,.,1,0,2
3,18,8.8812,0,.,1,0,2
4,1,12.0010,0,.,1,0,1
4,2,11.2543,0,.,1,0,1
4,4,9.3634,0,.,1,0,1
4,8,8.9268,0,.,1,0,1
4,11,10.1844,0,.,1,0,2
4,12,9.1643,0,.,1,0,2
4,14,9.4072,0,.,1,0,2
4,18,9.2523,0,.,1,0,2
5,1,8.0979,0,.,1,0,1
5,2,6.8496,0,.,1,0,1
5,4,6.4833,0,.,1,0,1
5,8,5.3018,0,.,1,0,1
5,11,4.8261,0,.,1,0,2
5,12,4.8009,0,.,1,0,2
5,14,4.0968,0,.,1,0,2
5,18,3.9619,0,.,1,0,2
6,1,7.6151,0,.,1,0,1
6,2,6.4948,0,.,1,0,1
6,4,4.9818,0,.,1,0,1
6,8,4.4421,0,.,1,0,1
6,11,3.4438,0,.,1,0,2
6,12,3.1916,0,.,1,0,2
6,14,3.1787,0,.,1,0,2
6,18,3.2352,0,.,1,0,2
7,1,3.7837,0,.,1,0,1
7,2,2.9795,0,.,1,0,1
7,4,1.9927,0,.,1,0,1
7,8,1.2329,0,.,1,0,1
7,11,-0.9294,0,.,1,0,2
7,12,-1.0659,0,.,1,0,2
7,14,-1.1358,0,.,1,0,2
7,18,-1.3142,0,.,1,0,2
8,1,10.0492,0,.,1,0,1
8,2,9.5782,0,.,1,0,1
8,4,8.3832,0,.,1,0,1
8,8,8.6782,0,.,1,0,1
8,11,5.8445,0,.,1,0,2
8,12,6.5344,0,.,1,0,2
8,14,6.1905,0,.,1,0,2
8,18,6.2090,0,.,1,0,2
";

/// The Emax model with `n_pad` zero-effect θ, estimated or FIX.
fn make_model(n_pad: usize, fix: bool) -> String {
    let pads: String = (0..n_pad)
        .map(|i| {
            if fix {
                format!("  theta PAD{i}(0.5, FIX)\n")
            } else {
                format!("  theta PAD{i}(0.5, 0.01, 10.0)\n")
            }
        })
        .collect();
    let terms: String = (0..n_pad).map(|i| format!(" + 0.0 * PAD{i}")).collect();
    format!(
        "[parameters]
  theta TVE0(10.0, 0.1, 100.0)
  theta TVEMAX(6.0, 0.1, 100.0)
  theta TVET50(2.0, 0.01, 100.0)
{pads}  omega ETA_E0 ~ 0.09
  sigma PROP ~ 0.04 (sd)

[individual_parameters]
  E0   = TVE0 * exp(ETA_E0) * (1.0{terms})
  EMAX = TVEMAX
  ET50 = TVET50

[structural_model]
  y = E0 - EMAX * TIME / (ET50 + TIME)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = focei
  covariance = false
"
    )
}

/// Parse-time warnings and the start-value evaluation of one twin.
fn evaluate(n_pad: usize, fix: bool) -> (Vec<String>, FitResult) {
    let parsed = parse_full_model(&make_model(n_pad, fix)).expect("padded model parses");
    assert_eq!(
        parsed.model.n_theta + parsed.model.n_eta,
        4 + n_pad,
        "axis count of the padded model (FIX θ included)"
    );
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("emax.csv");
    std::fs::write(&data, DATA).unwrap();
    let pop = read_nonmem_csv(&data, None, None).expect("data loads");
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let r = fit(&parsed.model, &pop, &parsed.model.default_params, &options).expect("fit runs");
    (parsed.model.parse_warnings.clone(), r)
}

#[test]
fn compartment_free_route_ignores_the_old_axis_cap_and_fix_theta() {
    let twins = [
        ("20 estimated pads (24 axes)", evaluate(20, false)),
        ("21 estimated pads (25 axes)", evaluate(21, false)),
        ("21 FIX pads (25 axes)", evaluate(21, true)),
    ];
    let (_, (_, at_cap)) = &twins[0];
    assert!(at_cap.ofv.is_finite(), "OFV {}", at_cap.ofv);

    for (label, (parse_warnings, r)) in &twins {
        // The likelihood is the same at every pad count, so the twins agree bit for bit.
        assert_eq!(r.subjects.len(), at_cap.subjects.len(), "{label}");
        for (a, b) in at_cap.subjects.iter().zip(&r.subjects) {
            assert_eq!(a.id, b.id);
            assert!(a.ofv_contribution.is_finite(), "subject {}", a.id);
            assert_eq!(
                a.ofv_contribution, b.ofv_contribution,
                "{label}, subject {}: per-subject OFV vs 24 axes",
                a.id
            );
            assert_eq!(a.eta, b.eta, "{label}, subject {}: η̂ vs 24 axes", a.id);
        }
        assert_eq!(r.ofv, at_cap.ofv, "{label}: OFV vs 24 axes");

        assert_eq!(
            r.gradient_method_inner, "analytic (Dual2)",
            "{label}: inner"
        );
        assert_eq!(
            r.gradient_method_outer, "analytic (Dual2)",
            "{label}: outer"
        );
        assert_eq!(
            r.optimizer, "auto (nlopt_lbfgs)",
            "{label}: `auto` takes L-BFGS on an analytic outer gradient, not BOBYQA"
        );

        // No warning — at parse time or from the fit — names a cap, a finite-difference
        // route, or advises FIXing parameters: none of them would be true.
        let stale: Vec<&String> = parse_warnings
            .iter()
            .chain(&r.warnings)
            .filter(|w| {
                w.contains("axis cap")
                    || w.contains("finite-difference")
                    || w.contains("finite difference")
                    || w.contains("FIX the ones")
                    || w.contains("estimates 25")
            })
            .collect();
        assert!(
            stale.is_empty(),
            "{label}: unexpected warning(s): {stale:?}"
        );
    }
}
