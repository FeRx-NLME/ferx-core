//! #1907 end to end: a closed-form IOV model whose `[individual_parameters]` program spans
//! more than 24 dual axes (`n_theta + n_eta + n_kappa`) keeps the exact analytic gradient.
//!
//! Before #1907 the program evaluator returned `None` past 24 axes while the model-level
//! gate still admitted the model, so every subject ran a reconverged finite-difference outer
//! gradient, and a finite-difference inner one, under `analytic (Dual2)` labels — the one
//! signal being the outer-fallback warning. The fixture is `examples/warfarin_iov.ferx`
//! (3 θ, 3 η, 1 κ: 7 axes) padded with fixed thetas `P_i`, each entering `CL` at 0: 17 pads
//! is the last single-chunk program (24 axes), 18 is one past it.
//!
//! No NONMEM anchor: the fix creates no new numeric object — the 25-axis program is
//! evaluated in two θ-column chunks whose blocks are bit-identical to the one-chunk
//! evaluation (pinned in Tier 1, `sens::provider::tests::iov_program_*`). What this pins is
//! the wiring: a real fit of the 25-axis model reports no subject on finite differences.
//! Mutation: `iov_combined_derivs_dyn` returning `None` past 24 axes again — the pad18 fit
//! then warns that 10 of 10 subjects use finite-difference inner gradients and that 10 of
//! 10 could not be given the exact analytic outer gradient (63 s instead of 11 s, debug).
//!
//! Tier 2: `maxiter = 1` runs one outer gradient evaluation and stops. Not `0`, which is the
//! evaluation-only path: it computes no outer gradient, so there would be no fallback to see.

use std::path::Path;

fn padded(n_pad: usize) -> String {
    let base = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/warfarin_iov.ferx"),
    )
    .unwrap();
    let pads: String = (1..=n_pad)
        .map(|i| format!("  theta P{i}(0.0, -1.0, 1.0) FIX\n"))
        .collect();
    let sum: String = (1..=n_pad).map(|i| format!(" + P{i}")).collect();
    let text = base
        .replacen("  omega ETA_CL ~", &format!("{pads}  omega ETA_CL ~"), 1)
        .replacen(
            // The leading newline skips the header comment's copy of the line.
            "\n  CL = TVCL * exp(ETA_CL + KAPPA_CL)",
            &format!("\n  CL = TVCL * exp(ETA_CL + KAPPA_CL{sum})"),
            1,
        );
    format!("{text}  maxiter    = 1\n  checkpoint = false\n")
}

fn fit(dir: &Path, n_pad: usize) -> ferx_core::FitResult {
    let model_path = dir.join(format!("pad{n_pad}.ferx"));
    std::fs::write(&model_path, padded(n_pad)).unwrap();
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/warfarin_iov.csv");
    ferx_core::run_model_with_data(model_path.to_str().unwrap(), Some(data.to_str().unwrap()))
        .map(|(fit, _)| fit)
        .unwrap_or_else(|e| panic!("pad{n_pad} fit failed: {e}"))
}

#[test]
fn iov_program_one_past_the_axis_cap_keeps_the_analytic_gradient() {
    let dir = tempfile::tempdir().unwrap();
    let narrow = fit(dir.path(), 17);
    let wide = fit(dir.path(), 18);
    eprintln!(
        "#1907 OFV pad17 {:.10} pad18 {:.10}; pad18 warnings: {:?}",
        narrow.ofv, wide.ofv, wide.warnings
    );
    for (name, r) in [("pad17", &narrow), ("pad18", &wide)] {
        assert!(r.ofv.is_finite(), "{name}: OFV {}", r.ofv);
        // Fixture: every pad is read by the program, so the program really is that wide.
        assert!(
            !r.warnings.iter().any(|w| w.contains("but not referenced")),
            "{name}: a pad is not in the program: {:?}",
            r.warnings
        );
        assert_eq!(r.gradient_method_inner, "analytic (Dual2)", "{name}: inner");
        assert_eq!(r.gradient_method_outer, "analytic (Dual2)", "{name}: outer");
        assert!(
            !r.warnings.iter().any(|w| w
                .contains("could not be given the exact analytic outer gradient")
                || w.contains("finite-difference inner gradients")),
            "{name}: a subject fell back to finite differences: {:?}",
            r.warnings
        );
    }
}
