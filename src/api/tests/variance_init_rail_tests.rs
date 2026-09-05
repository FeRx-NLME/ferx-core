//! `check_variance_init_rails` — a **free** variance whose packed start lands on
//! the optimizer's own lower rail (#1229).
//!
//! Every assertion here names the regression it exists to catch. Three of them
//! guard against tests that *cannot fail*, in the shapes CLAUDE.md lists:
//!
//! * The rail straddle asserts the **straddle itself** (`pack_params` on either
//!   side of `-6.0`), so a change to the parser's `1e-8` floor or to
//!   `compute_bounds`' rail cannot quietly turn the 5e-6/7e-6 pair into two
//!   coordinates on the same side agreeing for the wrong reason.
//! * The on-the-rail case pins a variance whose packed value is **bit-exactly**
//!   `-6.0` (measured: `6.144212353328210e-6`, `packed == -6.0` is `true`), which
//!   is the only input that can tell `<=` from `<`.
//! * The `FIX` case is not a "no diagnostic" formality: `compute_bounds` pins a
//!   fixed coordinate at `lower == upper == packed`, so `packed <= lower` is
//!   *true* for it and only the `packed_fixed_mask` consult keeps it green.

use super::*;

/// A minimal one-compartment IV model whose `[parameters]` block is spliced in
/// whole, so each test declares exactly the variance shape it is about.
fn model_with_parameters(params_block: &str) -> CompiledModel {
    let src = format!(
        "[parameters]\n{params_block}\n\
         \n\
         [individual_parameters]\n\
         CL = TVCL * exp(ETA_CL)\n\
         V  = TVV\n\
         \n\
         [structural_model]\n\
         pk one_cpt_iv(cl=CL, v=V)\n\
         \n\
         [error_model]\n\
         DV ~ proportional(PROP_ERR)\n"
    );
    crate::parser::model_parser::parse_model_string(&src)
        .unwrap_or_else(|e| panic!("model must parse: {e}\n--- source ---\n{src}"))
}

/// The default `[parameters]` block with one `omega ETA_CL` line substituted in.
fn params_with_omega(omega_line: &str) -> String {
    format!(
        "  theta TVCL(5.0, 0.001, 100.0)\n\
         \x20 theta TVV(50.0, 0.1, 500.0)\n\
         \x20 {omega_line}\n\
         \x20 sigma PROP_ERR ~ 0.04\n"
    )
}

fn rails_for_omega(omega_line: &str) -> Vec<Diagnostic> {
    let model = model_with_parameters(&params_with_omega(omega_line));
    check_variance_init_rails(&model.default_params)
}

/// The packed value of the sole Ω diagonal coordinate, straight from
/// `pack_params` — the quantity the check compares against the rail.
fn packed_omega_diagonal(omega_line: &str) -> f64 {
    let model = model_with_parameters(&params_with_omega(omega_line));
    packed_omega_diagonals(&model.default_params)[0]
}

/// Every Ω / Ω_IOV / mixture-Ω **diagonal** packed coordinate, in packed order.
fn packed_omega_diagonals(p: &ModelParameters) -> Vec<f64> {
    use crate::estimation::parameterization::{coordinate_kinds, pack_params, PackedCoordKind};
    let packed = pack_params(p);
    coordinate_kinds(p)
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == PackedCoordKind::OmegaDiagonal)
        .map(|(i, _)| packed[i])
        .collect()
}

/// The `-6.0` lower rail `compute_bounds` puts under every Ω diagonal, read
/// from `compute_bounds` rather than written as a literal, so the straddle
/// assertions below track the production bound.
fn omega_diagonal_rail(p: &ModelParameters) -> f64 {
    use crate::estimation::parameterization::{compute_bounds, coordinate_kinds, PackedCoordKind};
    let bounds = compute_bounds(p);
    let i = coordinate_kinds(p)
        .iter()
        .position(|k| *k == PackedCoordKind::OmegaDiagonal)
        .expect("model must have an omega diagonal");
    bounds.lower[i]
}

// ── the exact-zero declaration ──────────────────────────────────────────────

/// Regression: the shape #1227 shipped by accident and #1229 is about. A free
/// `~ 0.0` must be rejected, and the message must name the eta and the keyword
/// the user has to type.
#[test]
fn free_zero_omega_is_rejected_naming_the_eta_and_fix() {
    let diags = rails_for_omega("omega ETA_CL ~ 0.0");
    assert_eq!(diags.len(), 1, "{diags:#?}");
    let d = &diags[0];
    assert_eq!(d.code, "E_OMEGA_INIT_AT_RAIL");
    assert!(d.is_error(), "must be error severity, not a warning");
    assert_eq!(d.block.as_deref(), Some("parameters"));
    assert!(d.message.contains("ETA_CL"), "{}", d.message);
    assert!(d.message.contains("`FIX`"), "{}", d.message);
    assert!(d.message.contains("omega"), "{}", d.message);
    // The NONMEM refusal is quoted verbatim so a user who has seen NM-TRAN
    // error 76 recognises the same rejection.
    assert!(
        d.message
            .contains("INITIAL ESTIMATE OF VARIANCE CANNOT BE ZERO UNLESS FIXED"),
        "{}",
        d.message
    );
    assert!(
        d.suggestion.as_deref().unwrap_or_default().contains("FIX"),
        "{:?}",
        d.suggestion
    );
}

/// Regression: the `packed_fixed_mask` consult. `compute_bounds` pins a FIX-ed
/// coordinate at `lower == upper == packed`, so the bare predicate
/// `packed <= lower` is **true** here — dropping the mask check would reject the
/// one spelling the diagnostic tells users to write.
#[test]
fn fixed_zero_omega_is_accepted_even_though_it_sits_on_its_pinned_bound() {
    use crate::estimation::parameterization::{compute_bounds, pack_params};
    let model = model_with_parameters(&params_with_omega("omega ETA_CL ~ 0.0 FIX"));
    let p = &model.default_params;

    // The trap this test exists for, asserted rather than described: the FIX-ed
    // coordinate really is at its own lower bound.
    let packed = pack_params(p);
    let lower = compute_bounds(p).lower;
    let i = crate::estimation::parameterization::coordinate_kinds(p)
        .iter()
        .position(|k| *k == crate::estimation::parameterization::PackedCoordKind::OmegaDiagonal)
        .unwrap();
    assert!(
        packed[i] <= lower[i],
        "FIX pins lower == packed, so the bare predicate must fire here: \
         packed {} vs lower {}",
        packed[i],
        lower[i]
    );

    assert!(
        check_variance_init_rails(p).is_empty(),
        "{:#?}",
        check_variance_init_rails(p)
    );
}

// ── the rail straddle ───────────────────────────────────────────────────────

/// Regression: the predicate is `packed <= lower`, not "the declared variance
/// is zero". `5e-6` is a perfectly ordinary-looking number that packs *below*
/// the rail; `7e-6` — 1.4× larger — packs above it and fits fine.
///
/// The straddle is asserted on `pack_params` itself so this pair cannot become
/// a tautology: if the floor or the rail moved and both landed on the same side,
/// the two "one errors, one does not" assertions could still be satisfied by an
/// implementation that read the declared variance, but the straddle would fail.
#[test]
fn tiny_variances_straddle_the_minus_six_rail() {
    let model = model_with_parameters(&params_with_omega("omega ETA_CL ~ 5e-6"));
    let rail = omega_diagonal_rail(&model.default_params);
    assert_eq!(rail, -6.0, "the Ω-diagonal lower rail moved");

    let below = packed_omega_diagonal("omega ETA_CL ~ 5e-6");
    let above = packed_omega_diagonal("omega ETA_CL ~ 7e-6");
    assert!(
        below <= rail && above > rail,
        "5e-6 and 7e-6 must straddle the rail {rail}: packed {below} and {above}"
    );

    let rejected = rails_for_omega("omega ETA_CL ~ 5e-6");
    assert_eq!(rejected.len(), 1, "{rejected:#?}");
    assert_eq!(rejected[0].code, "E_OMEGA_INIT_AT_RAIL");
    // Message variant B: the value is above the regularisation floor, so it is
    // quoted and the remedy leads with a bigger start rather than with `FIX`.
    assert!(
        rejected[0].message.contains("5e-6"),
        "{}",
        rejected[0].message
    );
    assert!(
        rejected[0].message.contains("1e-5"),
        "{}",
        rejected[0].message
    );
    assert!(
        !rejected[0].message.contains("CANNOT BE ZERO"),
        "a non-zero start must not be reported as NONMEM's zero-variance error: {}",
        rejected[0].message
    );

    assert!(
        rails_for_omega("omega ETA_CL ~ 7e-6").is_empty(),
        "{:#?}",
        rails_for_omega("omega ETA_CL ~ 7e-6")
    );
}

/// Regression: `<=` mutated to `<`. `6.144212353328210e-6` is `e⁻¹²` to full
/// `f64` precision, so its packed value is **bit-exactly** `-6.0` — the single
/// input on which the two comparisons disagree. Measured, not assumed: the
/// literal `6.14421235e-6` from the issue text packs to `-6.000000000270841`,
/// which is strictly below the rail and would survive the mutation.
#[test]
fn variance_exactly_on_the_rail_is_rejected() {
    let packed = packed_omega_diagonal("omega ETA_CL ~ 6.144212353328210e-6");
    assert_eq!(
        packed, -6.0,
        "this literal must pack bit-exactly onto the rail, or it cannot tell \
         `<=` from `<`"
    );

    let diags = rails_for_omega("omega ETA_CL ~ 6.144212353328210e-6");
    assert_eq!(diags.len(), 1, "{diags:#?}");
    assert_eq!(diags[0].code, "E_OMEGA_INIT_AT_RAIL");
}

/// The other half of the differential pair: an ordinary variance is untouched,
/// so the check is not simply rejecting every Ω it sees.
#[test]
fn ordinary_variance_is_accepted() {
    assert!(
        rails_for_omega("omega ETA_CL ~ 0.09").is_empty(),
        "{:#?}",
        rails_for_omega("omega ETA_CL ~ 0.09")
    );
}

// ── the eigen-floor, pinned ─────────────────────────────────────────────────

/// Regression: the message-variant split. A declared `0.0` does not survive the
/// parse — `OmegaMatrix::from_matrix_with_mask` regularises the non-PD matrix by
/// `1e-8` — so the "add `FIX`" variant keys on the variance having landed at that
/// floor. If the floor moves, this reddens here rather than silently
/// re-labelling every free zero as a tiny-non-zero start.
#[test]
fn declared_zero_variance_arrives_at_the_regularisation_floor() {
    let model = model_with_parameters(&params_with_omega("omega ETA_CL ~ 0.0"));
    let om = &model.default_params.omega;
    assert_eq!(om.matrix[(0, 0)], 1e-8, "eigenvalue floor moved");
    assert_eq!(
        om.chol[(0, 0)],
        1e-4,
        "Cholesky of the floored variance moved"
    );
}

// ── block omega ─────────────────────────────────────────────────────────────

/// Regression: the predicate read from the declared variance rather than from
/// the Cholesky factor. In a `block_omega` the diagonal `L_ii` depends on the
/// off-diagonals, so only the packed factor says which coordinate is clamped —
/// and only the offending eta may be named.
#[test]
fn block_omega_with_one_zero_diagonal_names_only_that_eta() {
    let src = "  theta TVCL(5.0, 0.001, 100.0)\n\
               \x20 theta TVV(50.0, 0.1, 500.0)\n\
               \x20 block_omega (ETA_CL, ETA_V) = [0.0, 0.0, 0.04]\n\
               \x20 sigma PROP_ERR ~ 0.04\n";
    let model_src = format!(
        "[parameters]\n{src}\n\
         \n\
         [individual_parameters]\n\
         CL = TVCL * exp(ETA_CL)\n\
         V  = TVV * exp(ETA_V)\n\
         \n\
         [structural_model]\n\
         pk one_cpt_iv(cl=CL, v=V)\n\
         \n\
         [error_model]\n\
         DV ~ proportional(PROP_ERR)\n"
    );
    let model = crate::parser::model_parser::parse_model_string(&model_src)
        .unwrap_or_else(|e| panic!("block model must parse: {e}"));

    let diags = check_variance_init_rails(&model.default_params);
    assert_eq!(diags.len(), 1, "{diags:#?}");
    assert!(diags[0].message.contains("ETA_CL"), "{}", diags[0].message);
    // `ETA_V` (variance 0.04) is fine, and the block's off-diagonal — packed at
    // 0.0, a long way inside its own ±10 bound — is not a variance at all.
    assert!(
        !diags[0].message.contains("ETA_V"),
        "only the offending eta may be named: {}",
        diags[0].message
    );
}

/// Regression — the one the zero-diagonal case above cannot catch: reading the
/// **declared variance** instead of the packed Cholesky factor.
///
/// Both declared variances here are `0.09`, a long way inside the rail, so an
/// implementation that tested the declared value would report nothing. But the
/// block is correlated at ρ = 0.99997, and `L₂₂ = √(0.09 − L₂₁²) = 2.4e-3`
/// packs to `-6.01` — the coordinate the optimizer actually clamps. This is the
/// only fixture in the file where the declared variance and the factor
/// disagree about whether the model is on the rail.
#[test]
fn near_singular_block_flags_the_factor_not_the_declared_variance() {
    let model_src = "[parameters]\n\
         \x20 theta TVCL(5.0, 0.001, 100.0)\n\
         \x20 theta TVV(50.0, 0.1, 500.0)\n\
         \x20 block_omega (ETA_CL, ETA_V) = [0.09, 0.089997, 0.09]\n\
         \x20 sigma PROP_ERR ~ 0.04\n\
         \n\
         [individual_parameters]\n\
         CL = TVCL * exp(ETA_CL)\n\
         V  = TVV * exp(ETA_V)\n\
         \n\
         [structural_model]\n\
         pk one_cpt_iv(cl=CL, v=V)\n\
         \n\
         [error_model]\n\
         DV ~ proportional(PROP_ERR)\n";
    let model = crate::parser::model_parser::parse_model_string(model_src)
        .unwrap_or_else(|e| panic!("near-singular block must parse: {e}"));
    let p = &model.default_params;

    // The premise, asserted: every *declared* variance is interior, and it is
    // only the factor that is on the rail.
    assert_eq!(p.omega.matrix[(0, 0)], 0.09);
    assert_eq!(p.omega.matrix[(1, 1)], 0.09);
    let diagonals = packed_omega_diagonals(p);
    assert!(
        diagonals[0] > -6.0 && diagonals[1] <= -6.0,
        "L₁₁ must stay interior while L₂₂ lands on the rail: {diagonals:?}"
    );

    let diags = check_variance_init_rails(p);
    assert_eq!(diags.len(), 1, "{diags:#?}");
    assert!(diags[0].message.contains("ETA_V"), "{}", diags[0].message);
    // Variant B: the factor is above the regularisation floor, so this is not
    // reported as a declared zero.
    assert!(
        !diags[0].message.contains("CANNOT BE ZERO"),
        "{}",
        diags[0].message
    );
}

// ── Ω_IOV ───────────────────────────────────────────────────────────────────

/// Regression: the Ω_IOV segment skipped. `kappa` goes through the same
/// `build_omega_matrix` and the same `-6` rail, so a free `kappa ~ 0.0` is the
/// identical defect one segment further along the packed vector — and the
/// message must say `kappa`, not `omega`.
#[test]
fn free_zero_kappa_is_rejected_and_named_as_a_kappa() {
    let model_src = "[parameters]\n\
         \x20 theta TVCL(5.0, 0.001, 100.0)\n\
         \x20 theta TVV(50.0, 0.1, 500.0)\n\
         \x20 omega ETA_CL ~ 0.09\n\
         \x20 kappa KAPPA_CL ~ 0.0\n\
         \x20 sigma PROP_ERR ~ 0.04\n\
         \n\
         [individual_parameters]\n\
         CL = TVCL * exp(ETA_CL + KAPPA_CL)\n\
         V  = TVV\n\
         \n\
         [structural_model]\n\
         pk one_cpt_iv(cl=CL, v=V)\n\
         \n\
         [error_model]\n\
         DV ~ proportional(PROP_ERR)\n";
    let model = crate::parser::model_parser::parse_model_string(model_src)
        .unwrap_or_else(|e| panic!("IOV model must parse: {e}"));

    let diags = check_variance_init_rails(&model.default_params);
    assert_eq!(diags.len(), 1, "{diags:#?}");
    assert_eq!(diags[0].code, "E_OMEGA_INIT_AT_RAIL");
    assert!(
        diags[0].message.contains("KAPPA_CL"),
        "{}",
        diags[0].message
    );
    assert!(
        diags[0].message.contains("`kappa KAPPA_CL"),
        "an Ω_IOV coordinate must be reported as a `kappa` declaration: {}",
        diags[0].message
    );
    // `ETA_CL ~ 0.09` is interior; the BSV segment must not be swept in.
    assert!(!diags[0].message.contains("ETA_CL"), "{}", diags[0].message);
}

// ── mixture per-class Ω overrides (#977) ────────────────────────────────────

/// Regression: the mixture segment skipped. A `[mixture] omega(k)` override is
/// its own packed scalar carrying the base Ω diagonal's `-6` rail, so a free
/// zero there is trapped exactly like a base declaration — and the message has
/// to name the class, since `ETA_CL` itself is fine.
#[test]
fn free_zero_mixture_omega_override_is_rejected_naming_its_class() {
    let model_src = "[parameters]\n\
         \x20 theta TVCL1(1.0, 0.001, 100.0)\n\
         \x20 theta TVCL2(3.0, 0.001, 100.0)\n\
         \x20 theta TVV(50.0, 0.1, 500.0)\n\
         \x20 theta MIXL(0.0, -5.0, 5.0)\n\
         \x20 omega ETA_CL ~ 0.09\n\
         \x20 sigma PROP_ERR ~ 0.04\n\
         \n\
         [mixture]\n\
         nsub = 2\n\
         logit(1) = MIXL\n\
         omega(2) ETA_CL ~ 0.0\n\
         \n\
         [individual_parameters]\n\
         CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)\n\
         V  = TVV\n\
         \n\
         [structural_model]\n\
         pk one_cpt_iv(cl=CL, v=V)\n\
         \n\
         [error_model]\n\
         DV ~ proportional(PROP_ERR)\n";
    let model = crate::parser::model_parser::parse_model_string(model_src)
        .unwrap_or_else(|e| panic!("mixture model must parse: {e}"));

    let diags = check_variance_init_rails(&model.default_params);
    assert_eq!(diags.len(), 1, "{diags:#?}");
    assert_eq!(diags[0].code, "E_OMEGA_INIT_AT_RAIL");
    assert!(
        diags[0].message.contains("[mixture] omega(2)"),
        "the class has to be named — the base `omega ETA_CL ~ 0.09` is fine: {}",
        diags[0].message
    );
}

// ── Σ is out of scope ───────────────────────────────────────────────────────

/// Scope pin, measured not assumed: a free `sigma ~ 0.0 (sd)` is clamped onto
/// its own `-8` rail and *recovers the base optimum exactly* on the #1229
/// warfarin arm. The Σ rail does not trap, so including Σ "for symmetry" would
/// reject a declaration that works.
#[test]
fn free_zero_sigma_is_not_flagged() {
    let params = "  theta TVCL(5.0, 0.001, 100.0)\n\
                  \x20 theta TVV(50.0, 0.1, 500.0)\n\
                  \x20 omega ETA_CL ~ 0.09\n\
                  \x20 sigma PROP_ERR ~ 0.0 (sd)\n";
    let model = model_with_parameters(params);

    // The premise: Σ really is below its own rail, so "no diagnostic" is a
    // scope decision and not an accident of the fixture.
    use crate::estimation::parameterization::{
        compute_bounds, coordinate_kinds, pack_params, PackedCoordKind,
    };
    let p = &model.default_params;
    let packed = pack_params(p);
    let lower = compute_bounds(p).lower;
    let s = coordinate_kinds(p)
        .iter()
        .position(|k| *k == PackedCoordKind::Sigma)
        .expect("model must have a sigma");
    assert!(
        packed[s] <= lower[s],
        "the Σ coordinate must be on its rail for this scope pin to mean \
         anything: packed {} vs lower {}",
        packed[s],
        lower[s]
    );

    assert!(
        check_variance_init_rails(p).is_empty(),
        "{:#?}",
        check_variance_init_rails(p)
    );
}

// ── init_params, not default_params ─────────────────────────────────────────

/// Regression: the check reading `model.default_params` instead of the caller's
/// initial estimates. `--inits-from-nca` and the ferx-r override path both
/// replace the parsed inits, and it is the vector the optimizer starts from
/// that gets clamped.
#[test]
fn overriding_a_good_declaration_with_zero_is_rejected() {
    let model = model_with_parameters(&params_with_omega("omega ETA_CL ~ 0.09"));
    assert!(
        check_variance_init_rails(&model.default_params).is_empty(),
        "the declaration itself is fine"
    );

    let overridden = params_with_zero_omega_override(&model.default_params);
    let diags = check_variance_init_rails(&overridden);
    assert_eq!(diags.len(), 1, "{diags:#?}");
    assert_eq!(diags[0].code, "E_OMEGA_INIT_AT_RAIL");
}

/// The converse: a model declaring `~ 0.0` whose caller supplies a real
/// starting variance must fit. Without this the check could be reading the
/// declaration and passing the test above for the wrong reason.
#[test]
fn overriding_a_zero_declaration_with_a_real_variance_is_accepted() {
    let model = model_with_parameters(&params_with_omega("omega ETA_CL ~ 0.0"));
    assert_eq!(
        check_variance_init_rails(&model.default_params).len(),
        1,
        "the declaration itself is the rejected shape"
    );

    let good = model_with_parameters(&params_with_omega("omega ETA_CL ~ 0.09"));
    let mut overridden = model.default_params.clone();
    overridden.omega = good.default_params.omega.clone();
    assert!(
        check_variance_init_rails(&overridden).is_empty(),
        "{:#?}",
        check_variance_init_rails(&overridden)
    );
}

/// `ModelParameters` with the Ω rebuilt at a zero variance, the way a caller
/// handing `fit()` its own initial estimates would.
fn params_with_zero_omega_override(base: &ModelParameters) -> ModelParameters {
    let mut p = base.clone();
    p.omega = crate::types::OmegaMatrix::from_diagonal(&[0.0], p.omega.eta_names.clone());
    p
}

// ── n_eta = 0 ───────────────────────────────────────────────────────────────

/// A fixed-effects-only model (#989) has no Ω coordinates at all, so the walk
/// finds nothing. Deliberately *not* short-circuited on `n_eta == 0` in the
/// production code: a fast path there would make every other test in this file
/// exercise the guard instead of the predicate.
#[test]
fn fixed_effects_only_model_is_accepted() {
    let model_src = "[parameters]\n\
         \x20 theta TVCL(5.0, 0.001, 100.0)\n\
         \x20 theta TVV(50.0, 0.1, 500.0)\n\
         \x20 sigma PROP_ERR ~ 0.04\n\
         \n\
         [individual_parameters]\n\
         CL = TVCL\n\
         V  = TVV\n\
         \n\
         [structural_model]\n\
         pk one_cpt_iv(cl=CL, v=V)\n\
         \n\
         [error_model]\n\
         DV ~ proportional(PROP_ERR)\n";
    let model = crate::parser::model_parser::parse_model_string(model_src)
        .unwrap_or_else(|e| panic!("n_eta = 0 model must parse: {e}"));
    assert_eq!(model.n_eta, 0);
    assert!(
        check_variance_init_rails(&model.default_params).is_empty(),
        "{:#?}",
        check_variance_init_rails(&model.default_params)
    );
}

// ── several at once ─────────────────────────────────────────────────────────

/// One diagnostic per offending coordinate, so `ferx check` lists every
/// declaration the user has to edit rather than only the first.
#[test]
fn every_offending_coordinate_gets_its_own_diagnostic() {
    let params = "  theta TVCL(5.0, 0.001, 100.0)\n\
                  \x20 theta TVV(50.0, 0.1, 500.0)\n\
                  \x20 omega ETA_CL ~ 0.0\n\
                  \x20 omega ETA_V ~ 5e-6\n\
                  \x20 sigma PROP_ERR ~ 0.04\n";
    let model_src = format!(
        "[parameters]\n{params}\n\
         \n\
         [individual_parameters]\n\
         CL = TVCL * exp(ETA_CL)\n\
         V  = TVV * exp(ETA_V)\n\
         \n\
         [structural_model]\n\
         pk one_cpt_iv(cl=CL, v=V)\n\
         \n\
         [error_model]\n\
         DV ~ proportional(PROP_ERR)\n"
    );
    let model = crate::parser::model_parser::parse_model_string(&model_src)
        .unwrap_or_else(|e| panic!("two-eta model must parse: {e}"));

    let diags = check_variance_init_rails(&model.default_params);
    assert_eq!(diags.len(), 2, "{diags:#?}");
    assert!(diags[0].message.contains("ETA_CL"), "{}", diags[0].message);
    assert!(diags[1].message.contains("ETA_V"), "{}", diags[1].message);
    // Distinct variants: the zero gets the NONMEM quote, the 5e-6 does not.
    assert!(
        diags[0].message.contains("CANNOT BE ZERO"),
        "{}",
        diags[0].message
    );
    assert!(
        !diags[1].message.contains("CANNOT BE ZERO"),
        "{}",
        diags[1].message
    );
}
