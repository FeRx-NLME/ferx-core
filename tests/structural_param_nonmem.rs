//! NONMEM cross-check for the issue #261 acceptance model (FeRx-NLME/ferx-r#154).
//!
//! #261 made an undefined `[structural_model]` parameter reference a hard parse
//! error (covered by the fast unit tests in `parser/model_parser.rs`). This
//! guards the *other* half of that issue's acceptance: the **corrected** model —
//! the same benchmark with the missing `CL = KE * V` line restored — fits
//! cleanly and lands on the NONMEM objective rather than one of the two broken
//! regimes the issue surfaced.
//!
//! ## Provenance
//!
//! Model, data, and NONMEM control are Douglas Eleveld's exact reproduction from
//! FeRx-NLME/ferx-r#154 (the data is the PAGE blind-analysis "DATASIM" set —
//! 1-cpt oral, DV already on the log scale, hence `DV ~ log_additive`):
//!   - `MODEL_SRC` below is byte-identical to his `datsim.ferx`;
//!   - `data/datsim_oral.csv` is his `data.ferx.csv` (the `EVID` column is
//!     present, so this does not depend on the no-EVID dose inference in #262);
//!   - `tests/nonmem/datsim_oral.{ctl,lst}` is his NONMEM 7.6 control + output.
//!
//! ## NONMEM cross-check
//!
//! From `tests/nonmem/datsim_oral.lst` (NONMEM 7.6, FOCEI INTER, `$SIGMA 1`):
//!
//! | Parameter | ferx (FD) | NONMEM 7.6 |
//! |-----------|-----------|------------|
//! | OFV       | ~72.14    | 70.851 (WITHOUT CONSTANT) |
//! | TVV       | ~3.3      | 3.32       |
//! | TVKE      | ~-1.3     | -1.35      |
//! | TVKA      | ~-1.2     | -1.20      |
//!
//! ferx-core `main` reports OFV 71.28 on the autodiff path and ~72.14 on the FD
//! path (`--features ci`, the path this test runs under). The ~1.3-OFV ferx–
//! NONMEM gap — vs the near-exact match on well-conditioned data (cf.
//! `ltbs_convergence.rs`) — is this model's ill-conditioning: NONMEM itself only
//! reached `MINIMIZATION TERMINATED` (ωKA ≈ 93% CV). For that reason this test
//! does **not** assert `result.converged`; it only checks that the OFV lands in
//! a band that excludes the two broken regimes #261/#154 surfaced:
//!   - `CL` undefined → predictions floored → OFV ~2400 (now a parse error)
//!   - pre-Almquist FOCEI-INTER marginal → variance collapse → OFV ~ -29.5
//!
//! The band is centered on the FD value (what CI computes) and is two-sided ±3:
//! generous enough for FD/optimizer noise on this ill-conditioned surface, but
//! either broken regime is ~70+ OFV away and fails decisively. A one-sided
//! "lower is better" check (as in `gn_convergence.rs`) would wrongly *pass* the
//! -29.5 collapse. Re-anchor `EXPECTED_OFV` if a deliberate marginal-likelihood
//! change moves the minimum.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// The #154 benchmark model, corrected by restoring the `CL = KE * V` line whose
/// omission is the subject of #261. Byte-identical to Doug's `datsim.ferx`;
/// `[fit_options]` are set programmatically below.
const MODEL_SRC: &str = r"
[parameters]
  theta TVV (3.40, 1.61, 4.1)   # V  (log scale)
  theta TVKE(-1.2, -4.6, 0.7)   # KE (log scale)
  theta TVKA(-0.7, -4.6, 0.7)   # KA (log scale, excess over KE)

  block_omega (ETA_V, ETA_KE, ETA_KA) = [
    1.0,
    0.001, 1.0,
    0.001, 0.001, 1.0
  ]

  sigma ADD_ERR ~ 1.0           # match NONMEM $SIGMA 1

[individual_parameters]
  V  = exp(TVV  + ETA_V)
  KE = exp(TVKE + ETA_KE)
  KA = KE + exp(TVKA + ETA_KA)
  CL = KE * V                   # <-- the line whose omission is issue #261

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ log_additive(ADD_ERR)
";

/// FOCEI fit of the corrected #154 model must land on the NONMEM objective,
/// within a band that excludes both documented broken regimes.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored #261 / ferx-r#154 acceptance: opt in with --features slow-tests"
)]
fn corrected_structural_model_matches_nonmem_ofv() {
    // CI runs the FD path (`--features ci`); the ferx FD minimum on this model is
    // ~72.14 (ferx-r#154). NONMEM #OBJV (WITHOUT CONSTANT) is 70.851 and the ferx
    // autodiff minimum 71.28 — both within the band. See module docs for why the
    // band is two-sided, ±3, and why `converged` is not asserted.
    const EXPECTED_OFV: f64 = 72.14; // ferx FD path (what CI exercises)
    const TOLERANCE: f64 = 3.0;

    let model = parse_model_string(MODEL_SRC).expect("corrected #154 model must parse");
    let pop = read_nonmem_csv(Path::new("data/datsim_oral.csv"), None, None)
        .expect("datsim_oral data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.outer_maxiter = 300;
    opts.run_covariance_step = false;
    opts.verbose = false;

    let result =
        fit(&model, &pop, &model.default_params, &opts).expect("corrected #154 fit must run");

    assert!(
        result.ofv.is_finite(),
        "OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        (result.ofv - EXPECTED_OFV).abs() < TOLERANCE,
        "OFV {:.3} is outside the NONMEM-anchored band {:.2} ± {:.1} (NONMEM #OBJV \
         70.851, tests/nonmem/datsim_oral.lst); a value near -29.5 (marginal collapse) \
         or ~2400 (floored predictions) is the #261 / ferx-r#154 regression this guards",
        result.ofv,
        EXPECTED_OFV,
        TOLERANCE,
    );
}
