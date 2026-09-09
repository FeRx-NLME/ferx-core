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

/// The check as an ordinary fit would run it (`outer_maxiter = 500`).
///
/// The predicate tests below are about *which coordinates* are flagged, not
/// about the evaluation-only gate, so they all run with options that actually
/// optimise. The gate has its own tests further down.
fn rails_with_default_options(p: &ModelParameters) -> Vec<Diagnostic> {
    check_variance_init_rails(p, &FitOptions::default())
}

fn rails_for_omega(omega_line: &str) -> Vec<Diagnostic> {
    let model = model_with_parameters(&params_with_omega(omega_line));
    rails_with_default_options(&model.default_params)
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
        rails_with_default_options(p).is_empty(),
        "{:#?}",
        rails_with_default_options(p)
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
    // The quoted variance is `L²` reconstructed from the packed value — the
    // quantity actually compared against the rail. On a *diagonal* Ω that
    // equals the declared variance, which is why this coordinate can be checked
    // against the declaration at all; the block fixture is where they diverge.
    assert!(
        rejected[0].message.contains("5.000e-6"),
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

    let diags = rails_with_default_options(&model.default_params);
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

    let diags = rails_with_default_options(p);
    assert_eq!(diags.len(), 1, "{diags:#?}");
    let msg = &diags[0].message;
    assert!(msg.contains("ETA_V"), "{msg}");
    // Not reported as a declared zero: nothing here was regularised.
    assert!(!msg.contains("CANNOT BE ZERO"), "{msg}");

    // The assertion this fixture actually exists for (added after the #1246
    // review). The message-building code reads one of two quantities that
    // disagree *only here*, and the earlier assertions above were satisfied by
    // text that quoted the wrong one:
    //
    //   "initial variance 9e-2 for ETA_V (`omega ETA_V ~ 9e-2`) … every
    //    variance ≤ 6.14e-6 lands there … Start ETA_V at ≥ 1e-5"
    //
    // — self-contradictory (9e-2 is neither ≤ 6.14e-6 nor below 1e-5), naming a
    // line the user never wrote, and pointing away from the real cause. So:
    // never quote the natural-scale declared variance here, and never tell a
    // user whose variance is already 0.09 to raise it.
    assert!(
        !msg.contains("9e-2") && !msg.contains("0.09"),
        "the declared per-eta variance must not be quoted for a block: {msg}"
    );
    assert!(
        !msg.contains("≥ 1e-5"),
        "raising the declared variance is not the fix for a near-singular \
         block, and ETA_V is already far above 1e-5: {msg}"
    );
    // The real cause has to be named, since it is the only thing the user can
    // act on: the correlation, not either variance.
    assert!(
        msg.contains("correlation"),
        "a near-singular block must be reported as a correlation problem: {msg}"
    );
    // The keyword the user has to go and edit. `block_omega` and `block_kappa`
    // share this message arm, so the keyword is the *only* thing distinguishing
    // them — and every other assertion here passes with either one, which is how
    // a hardcoded `block_omega` survived into the kappa message unnoticed.
    assert!(
        msg.contains("`block_omega`") && !msg.contains("block_kappa"),
        "a block_omega must be named as one, not as the other block keyword: {msg}"
    );
    // Whatever quantity *is* quoted must be the one compared against the rail.
    let l22 = p.omega.chol[(1, 1)];
    assert!(
        msg.contains(&format!("{:.3e}", l22 * l22)),
        "the message must quote L₂₂² = {:.3e}, the variance on the clamped \
         coordinate: {msg}",
        l22 * l22
    );
    assert_eq!(diags[0].block.as_deref(), Some("parameters"));
}

/// An **indefinite** block declared with zero diagonals. Regression for the
/// second half of the `1e-8` misattribution: `reg` is `-min_eig + 1e-8` here,
/// not `1e-8`, so the regularised diagonal lands far above the floor and the
/// "you wrote 0.0" inference — which is only sound for a diagonal Ω — must not
/// fire. Before the review fix this rendered as
/// "initial variance 5.0000010000000004e-2 … packs to ln(L) = -8.86".
#[test]
fn indefinite_block_is_not_reported_as_a_declared_zero() {
    let model_src = "[parameters]\n\
         \x20 theta TVCL(5.0, 0.001, 100.0)\n\
         \x20 theta TVV(50.0, 0.1, 500.0)\n\
         \x20 block_omega (ETA_CL, ETA_V) = [0.0, 0.05, 0.0]\n\
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
        .unwrap_or_else(|e| panic!("indefinite block must parse: {e}"));
    let p = &model.default_params;

    // The premise: `reg` was *not* the 1e-8 floor, so a floor test cannot see
    // the declared zero here.
    assert!(
        p.omega.matrix[(0, 0)] > 1e-8,
        "an indefinite matrix regularises by -min_eig + 1e-8, well above the \
         floor: {}",
        p.omega.matrix[(0, 0)]
    );

    let diags = rails_with_default_options(p);
    assert!(!diags.is_empty(), "the block is still on the rail");
    for d in &diags {
        assert!(
            !d.message.contains("CANNOT BE ZERO"),
            "the declared value is not recoverable here, so NM-TRAN's \
             zero-variance refusal must not be quoted: {}",
            d.message
        );
        // No raw f64 debug dumps in user-facing text.
        assert!(!d.message.contains("5.0000010000000004"), "{}", d.message);
    }
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

    let diags = rails_with_default_options(&model.default_params);
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

    let diags = rails_with_default_options(&model.default_params);
    assert_eq!(diags.len(), 1, "{diags:#?}");
    assert_eq!(diags[0].code, "E_OMEGA_INIT_AT_RAIL");
    let d = &diags[0];
    assert!(
        d.message.contains("omega(2)"),
        "the class has to be named — the base `omega ETA_CL ~ 0.09` is fine: {}",
        d.message
    );

    // Both added after the #1246 review; the class-name assertion above was
    // satisfied while each of these was wrong.
    //
    // The block. `validate_model_file` turns this into the header line
    // `ferx check` prints and the JSON report carries, so stamping a `[mixture]`
    // declaration as `parameters` sends the reader to a block whose own
    // `omega ETA_CL ~ 0.09` line is perfectly fine.
    assert_eq!(
        d.block.as_deref(),
        Some("mixture"),
        "a per-class Ω override is declared in [mixture]"
    );
    // The suggestion has to be a line that parses. `FIX` is supported on a
    // mixture override, but the declaration is `omega(2) ETA_CL ~ 0.0 FIX`
    // *inside* the block — `[mixture] omega(2) …` is not valid syntax.
    let sugg = d.suggestion.as_deref().unwrap_or_default();
    assert!(
        sugg.contains("`omega(2) ETA_CL ~ 0.0 FIX`"),
        "the remedy must be spelled as it parses: {sugg}"
    );
    assert!(
        !sugg.contains("[mixture] omega"),
        "`[mixture] omega(2) …` is not a line that parses: {sugg}"
    );
    // `coordinate_names` displays this packed slot as `ETA_CL_MIX2`, which is
    // not a name anywhere in the model file. Neither the declaration echo nor
    // the prose may use it — a user grepping for it finds nothing.
    assert!(
        !d.message.contains("ETA_CL_MIX2") && !sugg.contains("ETA_CL_MIX2"),
        "the packed coordinate's display name must not reach the user:\n{}\n{sugg}",
        d.message
    );
}

// ── evaluation-only runs are out of scope (#1246 review, items 1 & 2) ───────

/// Regression: the check fired on runs where **no optimizer searches**, so
/// nothing is ever clamped and the rejection has no basis.
///
/// `outer_maxiter = 0` is NONMEM `MAXEVAL=0`: `optimize_population`
/// short-circuits to `evaluate_at_initial_params` before an optimizer is even
/// constructed. Two real callers land here — `ferx gam --no-fit`, and
/// `ferx-tools`' bootstrap `--dofv`, which re-evaluates each replicate at its
/// **own estimates**. A replicate that collapsed onto the rail comes back with
/// variance `exp(-12)`, which is a legitimate result rather than a declaration;
/// rejecting it dropped exactly the replicates the ΔOFV distribution exists to
/// characterise, and the caller maps the error to `None`, so it was silent.
#[test]
fn evaluation_only_run_is_not_judged() {
    let model = model_with_parameters(&params_with_omega("omega ETA_CL ~ 0.0"));
    let p = &model.default_params;

    let fitting = FitOptions::default();
    assert!(fitting.outer_maxiter > 0, "premise: the default fits");
    assert_eq!(
        check_variance_init_rails(p, &fitting).len(),
        1,
        "a fitting run must still reject it"
    );

    let eval_only = FitOptions {
        outer_maxiter: 0,
        ..FitOptions::default()
    };
    assert!(
        check_variance_init_rails(p, &eval_only).is_empty(),
        "nothing searches at outer_maxiter = 0, so no start can be trapped on \
         the rail: {:#?}",
        check_variance_init_rails(p, &eval_only)
    );
}

/// The knife-edge under the bootstrap finding, measured rather than reasoned.
///
/// The `#1246` review argued that `--dofv` drops railed replicates, but flagged
/// the round-trip as unverified: an optimizer that clamps a Ω diagonal onto the
/// `-6` rail reports `variance = exp(-12)`, and `params_from_estimates` sends
/// that back through `variance → matrix → chol → ln(L)`. Whether it lands on or
/// just above `-6.0` decides whether the replicate is dropped at all.
///
/// It lands **bit-exactly on** it — so `packed <= lower` holds and every railed
/// replicate would have been rejected, not just those that rounded down. That
/// makes the eval-only gate above load-bearing for `--dofv` rather than a
/// nicety, which is why this is pinned here: if the round-trip ever stopped
/// being exact, the gate would look unnecessary.
#[test]
fn a_railed_estimate_repacks_exactly_onto_the_rail() {
    use crate::estimation::parameterization::{compute_bounds, pack_params};

    // What an optimizer clamped to the lower bound reports back: L = exp(-6),
    // so the variance it writes into `FitResult.estimates` is exp(-6)² =
    // exp(-12). Rebuilt the way `params_from_estimates` rebuilds it.
    let railed = (-6.0f64).exp() * (-6.0f64).exp();
    let base = model_with_parameters(&params_with_omega("omega ETA_CL ~ 0.09"));
    let mut p = base.default_params.clone();
    p.omega = crate::types::OmegaMatrix::from_diagonal(&[railed], p.omega.eta_names.clone());

    let i = crate::estimation::parameterization::coordinate_kinds(&p)
        .iter()
        .position(|k| *k == crate::estimation::parameterization::PackedCoordKind::OmegaDiagonal)
        .unwrap();
    let packed = pack_params(&p)[i];
    let lower = compute_bounds(&p).lower[i];

    assert_eq!(
        packed, -6.0,
        "variance → chol → ln(L) must round-trip exactly onto the rail; got \
         {packed:.20}"
    );
    assert!(
        packed <= lower,
        "so a railed replicate satisfies the predicate: {packed} vs {lower}"
    );

    // Which is exactly why `--dofv` must not be judged: it re-enters `fit()`
    // with these estimates at `outer_maxiter = 0`.
    let eval_only = FitOptions {
        outer_maxiter: 0,
        ..FitOptions::default()
    };
    assert!(
        check_variance_init_rails(&p, &eval_only).is_empty(),
        "a railed replicate re-evaluated at maxiter = 0 must not be rejected"
    );
    // …and the same parameters *would* be rejected by a fitting run, so the
    // exemption is the gate and not something about these values.
    assert_eq!(
        check_variance_init_rails(&p, &FitOptions::default()).len(),
        1
    );
}

/// The other half of that gate: `outer_maxiter` is not the whole story. SAEM,
/// IMP, IMPMAP and Bayes carry their **own** iteration counts and never consult
/// it, so `outer_maxiter = 0` does not make them evaluation-only — and SAEM is
/// measured (#1229) to collapse a free zero exactly like FOCE. Without this the
/// gate would silently exempt four estimators.
#[test]
fn maxiter_zero_does_not_exempt_estimators_that_ignore_it() {
    let model = model_with_parameters(&params_with_omega("omega ETA_CL ~ 0.0"));
    let p = &model.default_params;

    for method in [
        EstimationMethod::Saem,
        EstimationMethod::Imp,
        EstimationMethod::Impmap,
        EstimationMethod::Bayes,
    ] {
        let opts = FitOptions {
            outer_maxiter: 0,
            method,
            ..FitOptions::default()
        };
        assert_eq!(
            check_variance_init_rails(p, &opts).len(),
            1,
            "{method:?} searches regardless of outer_maxiter, so the rail still \
             applies"
        );
    }

    // …and a method that *does* honour it stays exempt, so the two arms above
    // are a real discrimination rather than "everything is rejected".
    let foce_eval = FitOptions {
        outer_maxiter: 0,
        method: EstimationMethod::Foce,
        ..FitOptions::default()
    };
    assert!(check_variance_init_rails(p, &foce_eval).is_empty());
}

// ── what the message may claim about the declaration ────────────────────────

/// Regression: a declared `1e-9` was told it had written `~ 0.0` and that "a
/// zero variance is regularised to 1e-8" — a declaration it did not write, via
/// a mechanism that did not run. `[[1e-9]]` is positive-definite, so
/// `from_matrix_with_mask` takes the `Some(c)` arm and nothing is regularised;
/// the old `variance <= 1e-8` test swept it in anyway, producing text whose own
/// numbers disagreed (1e-8 packs to -9.21, but the message printed -10.36).
#[test]
fn tiny_positive_variance_is_not_reported_as_a_declared_zero() {
    let model = model_with_parameters(&params_with_omega("omega ETA_CL ~ 1e-9"));
    let p = &model.default_params;

    // The premise: it really is below the old floor test, and really was not
    // regularised.
    assert_eq!(p.omega.matrix[(0, 0)], 1e-9);
    assert!(1e-9 < 1e-8, "premise: below the regularisation floor");

    let diags = rails_with_default_options(p);
    assert_eq!(diags.len(), 1, "{diags:#?}");
    let msg = &diags[0].message;
    assert!(
        !msg.contains("CANNOT BE ZERO"),
        "NM-TRAN accepts 1e-9 and only refuses an exact zero, so quoting error \
         76 here is wrong: {msg}"
    );
    assert!(
        !msg.contains("regularised"),
        "no regularisation ran for a positive-definite declaration: {msg}"
    );
    assert!(msg.contains("ETA_CL"), "{msg}");
}

/// The differential partner: an **exact** hit on the regularisation floor on a
/// diagonal Ω is the one shape where quoting `~ 0.0` back, and citing NM-TRAN's
/// zero-variance refusal, are both sound — a positive declared variance is PD
/// on a diagonal Ω and survives verbatim, so only a zero lands exactly here.
#[test]
fn exact_floor_on_a_diagonal_omega_still_claims_the_zero() {
    let diags = rails_for_omega("omega ETA_CL ~ 0.0");
    assert_eq!(diags.len(), 1, "{diags:#?}");
    assert!(
        diags[0].message.contains("CANNOT BE ZERO"),
        "{}",
        diags[0].message
    );
}

/// `block_kappa` (IOV Option B) is correlated for exactly the same reason
/// `block_omega` is, so a near-singular one must get the same
/// correlation-shaped message rather than advice to raise a variance that is
/// already fine.
///
/// This test exists because a mutation exposed the gap: the predicate that was
/// *supposed* to separate block from diagonal sources was never reached for a
/// `block_kappa`, so making it always-true killed nothing. `block_kappa` now
/// takes the block arm, which makes that predicate provably redundant — it was
/// deleted rather than left as a second gate covering the first.
#[test]
fn near_singular_block_kappa_is_reported_as_a_correlation_problem() {
    let model_src = "[parameters]\n\
         \x20 theta TVCL(5.0, 0.001, 100.0)\n\
         \x20 theta TVV(50.0, 0.1, 500.0)\n\
         \x20 omega ETA_CL ~ 0.09\n\
         \x20 block_kappa (KAPPA_CL, KAPPA_V) = [0.09, 0.089997, 0.09]\n\
         \x20 sigma PROP_ERR ~ 0.04\n\
         \n\
         [individual_parameters]\n\
         CL = TVCL * exp(ETA_CL + KAPPA_CL)\n\
         V  = TVV * exp(KAPPA_V)\n\
         \n\
         [structural_model]\n\
         pk one_cpt_iv(cl=CL, v=V)\n\
         \n\
         [error_model]\n\
         DV ~ proportional(PROP_ERR)\n";
    let model = crate::parser::model_parser::parse_model_string(model_src)
        .unwrap_or_else(|e| panic!("block_kappa model must parse: {e}"));
    let p = &model.default_params;

    // The premise: Ω_IOV really is a block, and its second diagonal really is
    // on the rail while both declared variances are interior.
    let iov = p.omega_iov.as_ref().expect("model declares IOV");
    assert!(!iov.diagonal, "premise: block_kappa is not diagonal");
    assert_eq!(iov.matrix[(1, 1)], 0.09);

    let diags = rails_with_default_options(p);
    assert_eq!(diags.len(), 1, "{diags:#?}");
    let msg = &diags[0].message;
    assert!(
        msg.contains("correlation"),
        "a near-singular block_kappa is a correlation problem, not a small \
         variance: {msg}"
    );
    assert!(
        !msg.contains("≥ 1e-5"),
        "KAPPA_V is declared at 0.09; telling the user to raise it above 1e-5 \
         points away from the fix: {msg}"
    );
    // The regression this pair exists to catch, found by reading the diff after
    // the tests were already green: `block_omega` and `block_kappa` share this
    // message arm, and the arm hardcoded `block_omega`. Every assertion above
    // passed on a message telling a `block_kappa` user to go and fix a
    // `block_omega` — a block that appears nowhere in their file.
    assert!(
        msg.contains("`block_kappa`") && !msg.contains("block_omega"),
        "a block_kappa must be named as one, not as `block_omega`: {msg}"
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
        rails_with_default_options(p).is_empty(),
        "{:#?}",
        rails_with_default_options(p)
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
        rails_with_default_options(&model.default_params).is_empty(),
        "the declaration itself is fine"
    );

    let overridden = params_with_zero_omega_override(&model.default_params);
    let diags = rails_with_default_options(&overridden);
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
        rails_with_default_options(&model.default_params).len(),
        1,
        "the declaration itself is the rejected shape"
    );

    let good = model_with_parameters(&params_with_omega("omega ETA_CL ~ 0.09"));
    let mut overridden = model.default_params.clone();
    overridden.omega = good.default_params.omega.clone();
    assert!(
        rails_with_default_options(&overridden).is_empty(),
        "{:#?}",
        rails_with_default_options(&overridden)
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
        rails_with_default_options(&model.default_params).is_empty(),
        "{:#?}",
        rails_with_default_options(&model.default_params)
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

    let diags = rails_with_default_options(&model.default_params);
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
