//! `check_packed_start_in_box` — an initial estimate that packs **strictly**
//! outside its own box and is silently clamped onto it (#1251).
//!
//! Each test names the regression it exists to catch, and the mutation that
//! must redden it is recorded in the PR. Four shapes CLAUDE.md warns about are
//! handled explicitly here:
//!
//! * The equality case asserts the **mechanism** (`packed == bound`, so the
//!   clamp is a no-op) alongside the verdict, so widening either comparison to
//!   `<=` / `>=` reddens it for the stated reason rather than by coincidence.
//! * The `FIX` case likewise asserts `lower == upper == packed`, which is *why*
//!   a FIX-ed coordinate is never out of box — there is deliberately no mask
//!   consult in the predicate to mutate.
//! * The partition test asserts two **sets** of `(index, side)` — disjoint and
//!   together exhaustive over the out-of-box coordinates — not `len() == 1` on
//!   a one-coordinate fixture, which would stay green if a whole kind moved
//!   from one consumer to the other.
//! * The eval-only test asserts the straddle: the same start with
//!   `outer_maxiter = 0` and with `= 1`, so it cannot become a tautology if the
//!   predicate stops firing altogether.

use super::*;
use crate::estimation::parameterization::{
    coordinate_kinds, coordinates_outside_bounds, pack_params, pack_with_bounds, BoxSide,
    PackedCoordKind,
};

/// A minimal one-compartment IV model whose `[parameters]` block is spliced in
/// whole, so each test declares exactly the shape it is about. Mirrors the
/// sibling `variance_init_rail_tests.rs` helper.
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

/// The default `[parameters]` block with the `theta TVCL` line substituted in.
fn params_with_theta(theta_line: &str) -> String {
    format!(
        "  {theta_line}\n\
         \x20 theta TVV(50.0, 0.1, 500.0)\n\
         \x20 omega ETA_CL ~ 0.09\n\
         \x20 sigma PROP_ERR ~ 0.04\n"
    )
}

/// The default `[parameters]` block with the `omega ETA_CL` line substituted in.
fn params_with_omega(omega_line: &str) -> String {
    format!(
        "  theta TVCL(5.0, 0.001, 100.0)\n\
         \x20 theta TVV(50.0, 0.1, 500.0)\n\
         \x20 {omega_line}\n\
         \x20 sigma PROP_ERR ~ 0.04\n"
    )
}

/// The default `[parameters]` block with the `sigma PROP_ERR` line substituted in.
fn params_with_sigma(sigma_line: &str) -> String {
    format!(
        "  theta TVCL(5.0, 0.001, 100.0)\n\
         \x20 theta TVV(50.0, 0.1, 500.0)\n\
         \x20 omega ETA_CL ~ 0.09\n\
         \x20 {sigma_line}\n"
    )
}

/// A two-eta model, so a `block_omega` has an off-diagonal to be out of box on.
/// `model_with_parameters`' one-eta model cannot express one at all, which is
/// why the `OmegaOffDiagonal` arm had no fixture until #1309's review.
fn block_omega_model(omega_line: &str) -> CompiledModel {
    let src = format!(
        "[parameters]\n\
         \x20 theta TVCL(5.0, 0.001, 100.0)\n\
         \x20 theta TVV(50.0, 0.1, 500.0)\n\
         \x20 {omega_line}\n\
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
         DV ~ proportional(PROP_ERR)\n"
    );
    crate::parser::model_parser::parse_model_string(&src)
        .unwrap_or_else(|e| panic!("model must parse: {e}\n--- source ---\n{src}"))
}

/// The check as an ordinary fit would run it (`outer_maxiter` > 0, so the
/// eval-only exemption does not apply).
fn box_diags(params_block: &str) -> Vec<Diagnostic> {
    let model = model_with_parameters(params_block);
    check_packed_start_in_box(&model.default_params, &FitOptions::default())
}

fn only_message(diags: &[Diagnostic]) -> String {
    assert_eq!(
        diags.len(),
        1,
        "expected exactly one diagnostic, got {:#?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    diags[0].message.clone()
}

// ── T3: a θ strictly outside its own declared range is an error ─────────────

/// Regression: the shape #1251 is about, and the one
/// `docs/model-file/parameters.qmd` used to describe as untracked — a start
/// below its own declared lower bound is clamped onto it and the fit runs from
/// a value the user never wrote.
///
/// Three arms, each varying **one** thing against the same baseline: below the
/// lower, above the upper, and the identity-packed branch (a negative lower
/// bound opts out of log packing, so `compute_bounds` passes the declared
/// numbers through untouched and the comparison happens on a different scale).
#[test]
fn a_theta_start_strictly_outside_its_declared_range_is_an_error() {
    // Below the declared lower: `0.05 < 0.1`, and the clamp moves the start by
    // a factor of two — the plan's measured example.
    let below = box_diags(&params_with_theta("theta TVCL(0.05, 0.1, 10.0)"));
    let msg = only_message(&below);
    assert_eq!(below[0].code, "E_THETA_INIT_OUTSIDE_BOUNDS");
    assert!(
        below[0].is_error(),
        "a declared bound is the user's own: {msg}"
    );
    assert!(msg.contains("TVCL"), "{msg}");
    assert!(msg.contains("below"), "{msg}");

    // Above the declared upper — the same coordinate, one number changed.
    let above = box_diags(&params_with_theta("theta TVCL(500.0, 0.1, 10.0)"));
    assert_eq!(above.len(), 1, "{above:#?}");
    assert_eq!(above[0].code, "E_THETA_INIT_OUTSIDE_BOUNDS");
    assert!(above[0].is_error());
    assert!(above[0].message.contains("above"), "{}", above[0].message);

    // Identity packing: a negative lower bound opts out of `ln`, so the packed
    // coordinate *is* the declared value and the box *is* the declared range.
    // Without its own arm a predicate written only for the log branch passes.
    let ident = box_diags(&params_with_theta("theta TVCL(-5.0, -1.0, 10.0)"));
    assert_eq!(ident.len(), 1, "{ident:#?}");
    assert_eq!(ident[0].code, "E_THETA_INIT_OUTSIDE_BOUNDS");
    assert!(ident[0].is_error());
    assert!(ident[0].message.contains("below"), "{}", ident[0].message);

    // And the baseline the three vary from reports nothing at all.
    assert!(
        box_diags(&params_with_theta("theta TVCL(5.0, 0.1, 10.0)")).is_empty(),
        "an in-range start must be silent"
    );
}

// ── T16 (#1309 review round 2): a declared lower bound of `0` ──────────────

/// Regression: judging the declared question on the **packed** scale, which
/// `pack_with_bounds` has already made unanswerable.
///
/// `pack_params` floors the *value* at `THETA_PACK_FLOOR` and `unpinned_bounds`
/// floors the *bound* at the same constant, so `theta TVCL(-5.0, 0.0, 10.0)`
/// arrives with `packed == lower == ln(1e-10)` and the packed comparison sees a
/// start sitting exactly on its bound. Measured at `9ed38f6c`: **zero**
/// diagnostics, while the fit begins from `1e-10` — neither the declared `-5`
/// nor the declared `0`. NM-TRAN rejects `$THETA (0, -5, 10)` with the same
/// error 24 the message cites.
///
/// A declared lower of `0` is not a corner: `theta TVLAG(0.0, 0.0, 12.0)` is
/// one of T3b's own live fixtures and T3's identity arm is
/// `theta TVCL(-5.0, -1.0, 10.0)`, so the fixture set was literally one number
/// away from this case and never crossed the two.
///
/// The **premise** is asserted first — that the packed comparison really is
/// blind here — so this cannot quietly become a test of the packed predicate
/// if the flooring ever changes.
///
/// The differential control is the second arm: `theta TVCL(1e-12, 0.0, 10.0)`
/// is *inside* its declared range (`1e-12 >= 0`) and must stay silent, even
/// though it too is moved to `1e-10` by the floor. Without it a predicate that
/// simply reported every sub-floor θ would pass. (That second move is the
/// `1e-10` invisibility deferred to #1307 — a different object, and not this
/// error's.)
#[test]
fn a_theta_below_a_declared_lower_bound_of_zero_is_still_an_error() {
    let params = params_with_theta("theta TVCL(-5.0, 0.0, 10.0)");
    let model = model_with_parameters(&params);
    let p = &model.default_params;

    // Premise: the packed walk cannot see this, because both sides floor.
    let start = pack_with_bounds(p);
    assert_eq!(
        start.packed[0].to_bits(),
        start.bounds.lower[0].to_bits(),
        "premise: value and bound both floor to ln(1e-10), so the packed          comparison is an equality — packed {} vs lower {}",
        start.packed[0],
        start.bounds.lower[0],
    );
    let kinds = coordinate_kinds(p);
    assert_eq!(
        coordinates_outside_bounds(&start, &kinds).count(),
        0,
        "premise: nothing is outside the packed box"
    );

    // …and the declared question is answered anyway.
    let diags = box_diags(&params);
    let msg = only_message(&diags);
    assert_eq!(diags[0].code, "E_THETA_INIT_OUTSIDE_BOUNDS");
    assert!(diags[0].is_error(), "{msg}");
    assert!(msg.contains("starts at -5e0"), "{msg}");
    assert!(
        msg.contains("below its own declared lower bound of 0e0"),
        "{msg}"
    );
    // The message names where the fit really begins — the floor, not the
    // declared bound. `1e-10` formats as `1e-10` under `{:e}`.
    assert!(
        msg.contains("the fit begins from 1e-10"),
        "the start is clamped into the packed box, whose lower is the floor and          not the declared 0 — {msg}"
    );

    // Differential control: inside the declared range, and silent, although the
    // floor moves it just the same.
    assert!(
        box_diags(&params_with_theta("theta TVCL(1e-12, 0.0, 10.0)")).is_empty(),
        "1e-12 is inside [0, 10]; the floor moving it to 1e-10 is #1307's          object, not this error's"
    );
}

/// A **FIX**-ed θ outside its own declared range is not reported, and the
/// reason is not the reason it holds for every other coordinate.
///
/// T5 pins the Ω case, where the exclusion is structural: `pack_with_bounds`
/// sets `lower == upper == packed[i]`, so a strict inequality against the box
/// is false by construction and there is deliberately no mask consult to
/// mutate. `theta_outside_declared_range` never looks at that box — it compares
/// the declared numbers — so the pin is invisible to it and it needs an
/// explicit `theta_fixed` consult. Without one this fixture is refused with
/// `E_THETA_INIT_OUTSIDE_BOUNDS` while the fit would have run at exactly the
/// declared `0.05`, moving nothing: a false positive on a working model.
///
/// The premise is asserted first — the box really is pinned to the declared
/// value, so nothing is silently moved and there is nothing to report.
#[test]
fn a_fixed_theta_outside_its_declared_range_is_left_alone() {
    let params = params_with_theta("theta TVCL(0.05, 0.1, 10.0, FIX)");
    let model = model_with_parameters(&params);
    let p = &model.default_params;

    assert!(p.theta_fixed[0], "premise: the declaration is FIX-ed");
    let start = pack_with_bounds(p);
    assert_eq!(
        (
            start.bounds.lower[0].to_bits(),
            start.bounds.upper[0].to_bits()
        ),
        (start.packed[0].to_bits(), start.packed[0].to_bits()),
        "premise: the box is pinned to the packed start, so the fit runs at \
         the declared 0.05 and the clamp moves nothing"
    );

    assert!(
        box_diags(&params).is_empty(),
        "a FIX-ed θ is fitted at its declared value; refusing it would break a \
         model that works"
    );
    // And the un-FIX-ed twin — one token changed — *is* refused, so this test
    // cannot pass by the check having stopped firing.
    let free = box_diags(&params_with_theta("theta TVCL(0.05, 0.1, 10.0)"));
    assert_eq!(free.len(), 1, "{free:#?}");
    assert_eq!(free[0].code, "E_THETA_INIT_OUTSIDE_BOUNDS");
}

// ── T17 (#1309 review round 2): the remedy must be representable ───────────

/// Regression: advertising a range ferx cannot pack.
///
/// `theta TVCL(0.05, 0.1, 1e12)` used to suggest `inside (1e-1, 1e12)`; obeying
/// it with `1e10` then trips `W_INIT_OUTSIDE_BOUNDS` for exceeding the hidden
/// `1e9` cap, so the remedy sent the user from one diagnostic to another. Every
/// other arm already clamps its advice to `(THETA_PACK_FLOOR, THETA_PACK_CEIL)`.
///
/// Asserted as a **straddle**: the same declaration with an upper below the cap
/// keeps quoting the user's own number, so this cannot go green by the advice
/// always naming the cap.
#[test]
fn the_theta_remedy_never_advertises_a_bound_ferx_cannot_represent() {
    let capped = box_diags(&params_with_theta("theta TVCL(0.05, 0.1, 1e12)"));
    let suggestion = capped[0]
        .suggestion
        .clone()
        .unwrap_or_else(|| panic!("the error carries a remedy: {capped:#?}"));
    assert!(
        suggestion.contains("inside (1e-1, 1e9)"),
        "the advice must be the intersection of the declared range with what          the packer can represent — {suggestion}"
    );

    // Straddle: an upper *below* the cap is quoted verbatim.
    let uncapped = box_diags(&params_with_theta("theta TVCL(0.05, 0.1, 1e8)"));
    let suggestion = uncapped[0].suggestion.clone().unwrap();
    assert!(
        suggestion.contains("inside (1e-1, 1e8)"),
        "a declared upper the packer can represent is the user's own number —          {suggestion}"
    );
}

// ── T3b: exactly on a bound is left alone, and the test says why ────────────

/// Regression: an inclusive (`<=` / `>=`) rule. NM-TRAN rejects `init == bound`
/// too (errors 627 / 628), and ferx deliberately does not, because there the
/// clamp is a **no-op** — nothing is silently moved, which is this issue's whole
/// premise.
///
/// Both arms are shapes that live in this repo today: `theta TVF(1.0, 0.01,
/// 1.0)` (`crates/ferx-tools/src/iivsearch/mod_tests.rs`) and `theta TVLAG(0.0,
/// 0.0, 12.0)` (`tests/per_route_lag.rs`). An inclusive rule would reject both
/// to catch nothing.
///
/// Each arm asserts `packed == bound` **bit-exactly** as well as the verdict,
/// so the test states the mechanism. Without that it would pass for a predicate
/// that had simply stopped firing.
#[test]
fn a_theta_start_exactly_on_a_declared_bound_is_left_alone() {
    for (line, on_upper) in [
        ("theta TVCL(1.0, 0.01, 1.0)", true),
        ("theta TVCL(0.0, 0.0, 12.0)", false),
    ] {
        let params = params_with_theta(line);
        let model = model_with_parameters(&params);
        let p = &model.default_params;
        let start = pack_with_bounds(p);
        let bound = if on_upper {
            start.bounds.upper[0]
        } else {
            start.bounds.lower[0]
        };
        assert_eq!(
            start.packed[0].to_bits(),
            bound.to_bits(),
            "{line}: the premise is that the clamp is a no-op here — \
             packed {} vs bound {}",
            start.packed[0],
            bound
        );
        assert!(
            box_diags(&params).is_empty(),
            "{line}: nothing is moved, so there is nothing to report"
        );
    }
}

// ── T4: the hidden 1e9 cap is ferx's bound, not the user's ─────────────────

/// Regression: reporting a start that is **inside the user's declared box** as
/// if the user had got their own bounds wrong.
///
/// `theta TVCL(1e11, 0.001, 1e12)` is a perfectly consistent declaration.
/// `compute_bounds` substitutes its own `min(1e9)` cap for the declared upper,
/// so the packed start (`ln 1e11 = 25.33`) lands above the packed bound
/// (`ln 1e9 = 20.72`) — a ferx limit, hence a **warning**, and a message that
/// does not tell the user to widen a range they already widened.
#[test]
fn a_theta_start_past_the_hidden_upper_cap_is_a_warning_not_an_error() {
    let diags = box_diags(&params_with_theta("theta TVCL(1e11, 0.001, 1e12)"));
    let msg = only_message(&diags);
    assert!(
        !diags[0].is_error(),
        "the cap is ferx's, not the user's: {msg}"
    );
    assert_eq!(diags[0].code, "W_INIT_OUTSIDE_BOUNDS");
    assert!(
        msg.contains("internal"),
        "the message must say whose bound this is: {msg}"
    );
    assert!(
        !msg.contains("its own declared"),
        "must not blame the declaration: {msg}"
    );

    // The premise, stated rather than assumed: the start really is inside the
    // declared box, and it is the cap that excludes it.
    let model = model_with_parameters(&params_with_theta("theta TVCL(1e11, 0.001, 1e12)"));
    let p = &model.default_params;
    assert!(p.theta[0] < p.theta_upper[0], "inside the declared box");
    assert!(pack_params(p)[0] > pack_with_bounds(p).bounds.upper[0]);
}

// ── T5: the Ω upper rail, free and FIX-ed ──────────────────────────────────

/// Regression: #1229's predicate is `p > lo → continue`, i.e. **lower-rail
/// only**, so `omega ETA_CL ~ 1e8` packs at `+9.21` against the `+6` rail, is
/// clamped to a variance of `1.6e5`, and today says nothing at all.
///
/// A warning and not an error, measured: across all eight optimizer × method
/// arms on `examples/warfarin.ferx` the clamped start recovers the base
/// optimum (|ΔOFV| ≤ 0.14, θ and ω to 5–6 significant figures). See
/// `check_packed_start_in_box`'s doc comment.
#[test]
fn a_free_omega_start_above_the_upper_rail_is_reported_as_a_warning() {
    let diags = box_diags(&params_with_omega("omega ETA_CL ~ 1e8"));
    let msg = only_message(&diags);
    assert!(
        !diags[0].is_error(),
        "the rail recovers, so: warning — {msg}"
    );
    assert_eq!(diags[0].code, "W_INIT_OUTSIDE_BOUNDS");
    assert!(msg.contains("ETA_CL"), "{msg}");
    assert!(msg.contains("above"), "{msg}");
    // The rail variance the message quotes is exp(2·6) ≈ 1.63e5, not the
    // declared 1e8 — the coordinate that gets clamped, not the declaration.
    assert!(msg.contains("1.6"), "must quote the rail variance: {msg}");
}

/// Regression: a `FIX`-ed coordinate reported as out of box.
///
/// There is deliberately **no** `packed_fixed_mask` consult in
/// `coordinates_outside_bounds`, because `pack_with_bounds` pins a FIX-ed
/// coordinate to `lower == upper == packed[i]` from the same packed vector, so
/// a strict inequality is structurally false for it. A mask test would be a
/// second gate rejecting exactly what the first already rejects — the shape
/// that cannot fail. This test therefore asserts the **pin** as well as the
/// silence, so it pins the reason the consult is unnecessary.
#[test]
fn a_fixed_omega_start_is_never_out_of_box_because_the_box_is_pinned_to_it() {
    let params = params_with_omega("omega ETA_CL ~ 1e8 FIX");
    let model = model_with_parameters(&params);
    let p = &model.default_params;
    let start = pack_with_bounds(p);
    let i = coordinate_kinds(p)
        .iter()
        .position(|k| *k == PackedCoordKind::OmegaDiagonal)
        .expect("the model has one Ω diagonal");

    assert!(start.fixed[i], "premise: the coordinate is FIX-ed");
    assert_eq!(start.bounds.lower[i].to_bits(), start.packed[i].to_bits());
    assert_eq!(start.bounds.upper[i].to_bits(), start.packed[i].to_bits());
    assert!(
        box_diags(&params).is_empty(),
        "a pinned box cannot be strictly escaped"
    );
    // The same declaration *without* `FIX` is reported — so the silence above
    // is the pin, not the predicate having stopped working.
    assert_eq!(box_diags(&params_with_omega("omega ETA_CL ~ 1e8")).len(), 1);
}

// ── T6: Σ is a warning, and `fit()` is not refused ─────────────────────────

/// Regression: promoting Σ to an error "for symmetry" with #1229's Ω rail.
///
/// #1229 measured that a free `sigma ~ 0.0` clamped onto its `-8` rail
/// *recovers the base optimum exactly*, which is why the rail check excludes Σ
/// entirely. Both Σ rails stay warnings here for the same reason.
///
/// `first_error(..).is_ok()` is the exact predicate `fit()` applies, so this
/// pins "the fit is not refused" without running one.
#[test]
fn a_sigma_start_outside_either_rail_is_a_warning_that_does_not_refuse_the_fit() {
    // Four arms, and the last two are the point: `(sd)` keeps the declared
    // number, a plain `sigma X ~ v` declares the **variance** and
    // `model_parser` square-roots it, so the number the message quotes is an
    // SD that appears nowhere in the model file. `~ 1e6` reports `1.000e3`.
    // Round 1's review fixed exactly this trap on the `block_omega` arm (L vs
    // L²); Σ was the one arm that kept a scale-free noun (#1309 review).
    for (line, quoted) in [
        // packs to ln(1e-10) = -23.03, below -8
        ("sigma PROP_ERR ~ 0.0 (sd)", "an SD of 1.000e-10"),
        // packs to +13.8, above +5
        ("sigma PROP_ERR ~ 1e6 (sd)", "an SD of 1.000e6"),
        // the plain spelling: sqrt(1e6) = 1e3, and sqrt(1e-9) = 3.162e-5
        ("sigma PROP_ERR ~ 1e6", "an SD of 1.000e3"),
        ("sigma PROP_ERR ~ 1e-9", "an SD of 3.162e-5"),
    ] {
        let diags = box_diags(&params_with_sigma(line));
        let msg = only_message(&diags);
        assert!(
            !diags[0].is_error(),
            "{line}: Σ recovers from its rails — {msg}"
        );
        assert_eq!(diags[0].code, "W_INIT_OUTSIDE_BOUNDS");
        assert!(msg.contains("PROP_ERR"), "{msg}");
        // The number itself, so a change of scale or of format reddens this.
        // Nothing anywhere read this arm's number before (#1309 review
        // finding 3): `hit.packed.exp()` → `hit.packed` was a silent
        // regression across the whole corpus.
        assert!(msg.contains(quoted), "{line}: must quote {quoted} — {msg}");
        // …and say which scale it is on, since the plain spelling's declared
        // number is a variance.
        assert!(
            msg.contains("SD rail of"),
            "{line}: the rail is an SD too, and must say so — {msg}"
        );
        // The remedy quotes the interval `unpinned_bounds` actually pushed.
        // Before #1309's review it carried its own `(-8.0).exp()` / `5.0.exp()`
        // literals, so moving the rail would have left the message naming the
        // old one with nothing to catch it: these are `SIGMA_PACK_LOWER` and
        // `SIGMA_PACK_UPPER` read back, at three digits.
        assert!(
            msg.contains("at an SD inside (3.355e-4, 1.484e2)"),
            "{line}: the remedy must quote the rails the box uses, on the scale \
             they are on — {msg}"
        );
        assert!(
            crate::diagnostics::first_error(&diags).is_ok(),
            "{line}: `fit()` must not refuse this"
        );
    }
}

// ── T7: the eval-only exemption, asserted as a straddle ───────────────────

/// Regression: removing the `outer_search_runs` gate.
///
/// An eval-only run clamps too — it just does not *hold* the clamped value
/// through a search, which is the stickiness this check exists to warn about.
/// `tests/check_command.rs::an_eval_only_fit_reports_the_rail_not_the_declared_zero`
/// pins that `fit()` succeeds at `outer_maxiter = 0`, so an eval-only error
/// reddens it too.
///
/// Asserted as a **straddle** — the same start under `= 0` and under `= 1` —
/// so it cannot go quietly green by the predicate never firing at all.
#[test]
fn the_eval_only_exemption_is_a_straddle_not_a_blanket_silence() {
    let model = model_with_parameters(&params_with_theta("theta TVCL(0.05, 0.1, 10.0)"));
    let p = &model.default_params;

    let eval_only = FitOptions {
        outer_maxiter: 0,
        ..FitOptions::default()
    };
    let searching = FitOptions {
        outer_maxiter: 1,
        ..FitOptions::default()
    };
    assert!(
        check_packed_start_in_box(p, &eval_only).is_empty(),
        "an eval-only run is exempt"
    );
    assert_eq!(
        check_packed_start_in_box(p, &searching).len(),
        1,
        "and one outer iteration is enough to make it matter"
    );
}

// ── T8: the partition against `check_variance_init_rails` ─────────────────

/// Regression: a whole coordinate kind + side falling between the two checks,
/// or being claimed by both.
///
/// Asserted as **sets of `(index, side)`** — not as `diags.len() == 1` on a
/// single-coordinate fixture, which stays green when a kind moves from one
/// consumer to the other.
///
/// The property asserted is generic over the kinds, but the fixture is **not
/// exhaustive over them** and must not be read as though it were: it carries
/// θ-Below-declared, Ω-diagonal-Below, Ω-diagonal-Above and Σ-Below. Absent are
/// the Ω off-diagonal (T13), a `block_omega` diagonal (T14), Σ-Above, the θ
/// internal cap (T4), Ω_IOV and both mixture segments. Widening this fixture is
/// welcome; believing it already covers them is what to avoid (#1309 review).
///
/// The two claims must be **disjoint** (nothing double-reported) and together
/// **exhaustive** (nothing dropped) over `coordinates_outside_bounds`.
#[test]
fn the_two_start_checks_partition_the_out_of_box_coordinates() {
    // θ below its declared lower, Ω diagonal below the −6 rail (#1229's), Ω
    // diagonal above the +6 rail, and Σ below the −8 rail. `TVV` stays in range
    // so at least one θ is *not* claimed by either.
    let params = "  theta TVCL(0.05, 0.1, 10.0)\n\
                  \x20 theta TVV(50.0, 0.1, 500.0)\n\
                  \x20 omega ETA_CL ~ 0.0\n\
                  \x20 omega ETA_V ~ 1e8\n\
                  \x20 sigma PROP_ERR ~ 0.0 (sd)\n";
    let src = format!(
        "[parameters]\n{params}\n\
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
    let model = crate::parser::model_parser::parse_model_string(&src)
        .unwrap_or_else(|e| panic!("model must parse: {e}\n{src}"));
    let p = &model.default_params;
    let opts = FitOptions::default();

    let start = pack_with_bounds(p);
    let kinds = coordinate_kinds(p);
    let all: Vec<(usize, BoxSide)> = coordinates_outside_bounds(&start, &kinds)
        .map(|h| (h.index, h.side))
        .collect();
    // Four kinds+sides, or the fixture has stopped exercising the partition.
    // Four of the nine reachable combinations, not all of them — see the doc
    // comment.
    assert_eq!(
        all.len(),
        4,
        "fixture must carry the four kinds and sides it is built for: {all:?}"
    );

    // What #1229 claims: `packed <= lower` on the Ω / Ω_IOV / mixture-Ω
    // diagonals, i.e. the `Below` half of the out-of-box set on those.
    let rails = check_variance_init_rails(p, &opts);
    let boxed = check_packed_start_in_box(p, &opts);
    assert_eq!(
        rails.len(),
        1,
        "ETA_CL below its rail, and only that: {rails:#?}"
    );
    assert_eq!(
        boxed.len(),
        3,
        "TVCL below its declared lower, ETA_V above its rail, PROP_ERR below \
         the Σ rail: {boxed:#?}"
    );

    // Disjoint: no coordinate name appears in both sets.
    for r in &rails {
        for b in &boxed {
            let shared = ["TVCL", "ETA_CL", "ETA_V", "PROP_ERR"]
                .iter()
                .filter(|n| r.message.contains(**n) && b.message.contains(**n))
                .count();
            assert_eq!(
                shared, 0,
                "a coordinate is claimed twice:\n  rail: {}\n  box:  {}",
                r.message, b.message
            );
        }
    }
    // Exhaustive: every out-of-box coordinate is named by exactly one of them.
    let names = crate::estimation::parameterization::coordinate_names(p);
    for (i, side) in &all {
        let n = &names[*i];
        let in_rails = rails.iter().any(|d| d.message.contains(n.as_str()));
        let in_boxed = boxed.iter().any(|d| d.message.contains(n.as_str()));
        assert!(
            in_rails ^ in_boxed,
            "{n} ({side:?}) must be claimed by exactly one check \
             (rails: {in_rails}, box: {in_boxed})"
        );
    }
    // …and the partition falls where it is documented to: `Below` on a variance
    // diagonal to #1229, everything else here.
    assert_eq!(
        rails.len() + boxed.len(),
        all.len(),
        "the two claims together must count the whole out-of-box set"
    );
}

// ── T12: the blast-radius ratchet over every shipped model file ───────────

/// Regression: widening the predicate until it catches something ferx itself
/// ships, or adding an example whose start is silently clamped.
///
/// Named, not counted. A count would stay green if one fixture drifted out of
/// the box while another drifted in — this fails on the *identity* of the
/// coordinate, so either direction reddens.
///
/// The eleven entries below are all `OmegaDiagonal` / `Below`, i.e. all already
/// in #1229's scope: eight simulate-only adaptive-dosing models declaring
/// `omega ~ 1e-10`, two `reset_init_snapshot_*` anchors declaring `~ 0.0`, and
/// `tests/fixtures/iov_scaling.ferx` declaring `kappa KAPPA_V ~ 0.000001`.
/// **Zero** θ, Σ, off-diagonal or upper-rail hits — which is what makes this
/// change safe for everything shipped.
///
/// The walk is over the **whole tree**, not `examples/` + `nonmem_anchor/`.
/// Round 1's review verified the wider scope by hand and the committed test did
/// not inherit it: `.ferx` files also live in `tests/fixtures/`, `tests/nonmem/`,
/// `tests/reference/` and `tools/`, and `tests/fixtures/` is where the next
/// fixture lands. The eleventh entry is exactly what the narrower sweep missed
/// (#1309 review).
///
/// Both start-side predicates are ratcheted here, not just the out-of-box one.
/// `coordinates_with_inverted_bounds` is the newest arm, it is an **error**, and
/// it is the one arm with no `maxiter = 0` exemption — so it refuses a model in
/// strictly more situations than anything else in this file, and it shipped with
/// no corpus guard at all. The corpus carries **zero** inverted boxes; this is
/// what keeps that true.
#[test]
fn no_shipped_model_file_starts_outside_its_box_except_the_known_eleven() {
    let allowed: std::collections::BTreeSet<&str> = [
        "examples/adaptive_platelet_ladder.ferx::ETA_CL",
        "examples/adaptive_vanco_auc.ferx::ETA_CL",
        "examples/adaptive_vanco_iov.ferx::ETA_CL",
        "examples/adaptive_vanco_iov_loading.ferx::ETA_CL",
        "examples/adaptive_vanco_loading.ferx::ETA_CL",
        "examples/adaptive_vanco_renal.ferx::ETA_CL",
        "examples/adaptive_vanco_renal_iov.ferx::ETA_CL",
        "examples/adaptive_vanco_renal_loading.ferx::ETA_CL",
        "nonmem_anchor/reset_init_snapshot_fit.ferx::ETA_CL",
        "nonmem_anchor/reset_init_snapshot_occ.ferx::ETA_CL",
        "tests/fixtures/iov_scaling.ferx::KAPPA_V",
    ]
    .into_iter()
    .collect();

    // Every `.ferx` in the repository, `target/` and the git plumbing aside.
    fn collect(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let path = e.path();
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            if path.is_dir() {
                // `target/` holds build output, `.claude/worktrees/` holds
                // whole other checkouts of this repo — neither is this tree.
                if matches!(name.as_str(), "target" | ".git" | ".claude" | "_site") {
                    continue;
                }
                collect(&path, out);
            } else if path.extension().and_then(|s| s.to_str()) == Some("ferx") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    collect(std::path::Path::new("."), &mut files);
    files.sort();

    let mut found = std::collections::BTreeSet::new();
    let mut inverted = Vec::new();
    let mut declared = Vec::new();
    let mut parsed_files = 0usize;
    for path in &files {
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        // A file this build cannot parse (a feature-gated block) is not
        // evidence either way; the count below keeps that from silently
        // becoming "no files were checked".
        let Ok(model) = crate::parser::model_parser::parse_model_string(&src) else {
            continue;
        };
        parsed_files += 1;
        let p = &model.default_params;
        let start = pack_with_bounds(p);
        let kinds = coordinate_kinds(p);
        let names = crate::estimation::parameterization::coordinate_names(p);
        let file = path
            .strip_prefix("./")
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        for hit in coordinates_outside_bounds(&start, &kinds) {
            assert_eq!(
                (hit.kind, hit.side),
                (PackedCoordKind::OmegaDiagonal, BoxSide::Below),
                "{file}: a shipped model may only be out of box on the \
                 variance rail #1229 already owns, got {:?} {:?} on {}",
                hit.kind,
                hit.side,
                names[hit.index],
            );
            found.insert(format!("{file}::{}", names[hit.index]));
        }
        for hit in
            crate::estimation::parameterization::coordinates_with_inverted_bounds(&start, &kinds)
        {
            inverted.push(format!("{file}::{}", names[hit.index]));
        }
        // The declared-scale θ predicate is strictly stronger than the packed
        // one on the declared sides (#1309 review finding 1), so it gets its
        // own row rather than riding on the sweep above: a shipped θ outside
        // its declaration would be an *error* at fit time.
        declared.extend(
            crate::estimation::parameterization::theta_outside_declared_range(p)
                .map(|hit| format!("{file}::{} starts at {:e}", names[hit.index], hit.value)),
        );
    }

    // Guard against the walk finding nothing because the paths moved: this is a
    // positive count, so an empty sweep fails rather than passing vacuously.
    // 154 parsed at the time of writing, over 176 found.
    assert!(
        parsed_files >= 150,
        "expected the shipped model corpus, parsed only {parsed_files} files — \
         has the working directory or the layout changed?"
    );
    assert!(
        inverted.is_empty(),
        "no shipped model may have an empty optimizer box — \
         `E_INIT_BOUNDS_INVERTED` refuses these even at `maxiter = 0`: {inverted:?}"
    );
    assert!(
        declared.is_empty(),
        "no shipped model may start a theta outside its declared range — \
         `E_THETA_INIT_OUTSIDE_BOUNDS` refuses these: {declared:?}"
    );
    let found_refs: std::collections::BTreeSet<&str> = found.iter().map(String::as_str).collect();
    assert_eq!(
        found_refs, allowed,
        "the set of shipped out-of-box coordinates changed"
    );
}

// ── T15 (#1309 review): an EMPTY box, which used to abort the process ──────

/// Regression: `coordinates_outside_bounds` returning `None` for `lower >
/// upper` and nothing else claiming it.
///
/// That is not a benign skip. `clamp_to_bounds` calls `f64::clamp`, which
/// **panics** on `min > max`, and every optimizer entry clamps the start before
/// its first evaluation — so an empty box aborted the process with
/// `min > max, or either was NaN` from `core/src/num/f64.rs` instead of
/// producing a diagnostic. Measured on all three arms below at `d8ae4412`.
///
/// Three causes, one variable each, and the third is the one that makes this
/// more than a typo guard: `theta TVCL(1e-12, 1e-13, 1e-11)` is an ordinary
/// declaration of a small parameter, with its own lower bound below its own
/// upper, and it is ferx's `1e-10` packing floor — substituted for the declared
/// lower — that empties the box.
#[test]
fn a_theta_whose_packed_box_is_empty_is_an_error_not_a_panic() {
    for (line, cause) in [
        // The declaration itself is empty: bounds swapped.
        ("theta TVCL(1.0, 5.0, 2.0)", "declared range is empty"),
        // Legal declaration, entirely above ferx's 1e9 cap.
        ("theta TVCL(5e9, 2e9, 1e12)", "entirely above"),
        // Legal declaration, entirely below ferx's 1e-10 floor.
        ("theta TVCL(1e-12, 1e-13, 1e-11)", "entirely below"),
    ] {
        let params = params_with_theta(line);
        let model = model_with_parameters(&params);
        let p = &model.default_params;

        // The premise: the box really is empty. Asserted so the test cannot go
        // green by the predicate having stopped firing.
        let start = pack_with_bounds(p);
        assert!(
            start.bounds.lower[0] > start.bounds.upper[0],
            "{line}: premise — packed lower {} must exceed upper {}",
            start.bounds.lower[0],
            start.bounds.upper[0]
        );

        let diags = check_packed_start_in_box(p, &FitOptions::default());
        let msg = only_message(&diags);
        assert_eq!(diags[0].code, "E_INIT_BOUNDS_INVERTED");
        assert!(diags[0].is_error(), "{line}: {msg}");
        assert!(msg.contains("TVCL"), "{line}: {msg}");
        assert!(msg.contains("empty optimizer box"), "{line}: {msg}");
        assert!(
            msg.contains(cause),
            "{line}: the message must name *which* of the three causes, got {msg}"
        );
        // And `fit()` refuses, which is the whole point — `first_error` is the
        // exact predicate it applies.
        assert!(
            crate::diagnostics::first_error(&diags).is_err(),
            "{line}: the fit must stop here rather than in the clamp"
        );
    }

    // The differential half, on the arm closest to a working declaration: one
    // number moved so the box is non-empty, and the check goes silent.
    assert!(
        box_diags(&params_with_theta("theta TVCL(1e-12, 1e-13, 1e-9)")).is_empty(),
        "an upper bound above the 1e-10 floor leaves a representable box"
    );
}

/// The empty-box error is **not** exempt at `outer_maxiter = 0`, unlike every
/// other verdict this check ships.
///
/// The eval-only exemption exists because a clamped start does not *stick*
/// without a search. An empty box is not clamped at all — it panics — and
/// `evaluate_at_initial_params` clamps as well, so exempting it would leave
/// exactly the aborting case unreported. Asserted as a straddle against the
/// out-of-box error on the same fixture shape, which *is* exempt.
#[test]
fn the_empty_box_error_survives_the_eval_only_exemption() {
    let eval_only = FitOptions {
        outer_maxiter: 0,
        ..FitOptions::default()
    };

    let empty = model_with_parameters(&params_with_theta("theta TVCL(1.0, 5.0, 2.0)"));
    let d = check_packed_start_in_box(&empty.default_params, &eval_only);
    assert_eq!(d.len(), 1, "an empty box is reported even at maxiter = 0");
    assert_eq!(d[0].code, "E_INIT_BOUNDS_INVERTED");

    // The other side of the straddle: a merely out-of-box start on the same
    // model shape stays exempt, so this test pins the *difference* rather than
    // the exemption having been deleted wholesale.
    let outside = model_with_parameters(&params_with_theta("theta TVCL(0.05, 0.1, 10.0)"));
    assert!(
        check_packed_start_in_box(&outside.default_params, &eval_only).is_empty(),
        "a clamped-but-orderable start is still exempt at maxiter = 0"
    );
}

// ── T13 (#1309 review): the Ω off-diagonal arm ─────────────────────────────

/// Regression: the `PackedCoordKind::OmegaOffDiagonal` arm of
/// `check_packed_start_in_box`, which no test at any tier reached when this PR
/// was first reviewed. Replacing its body with `unreachable!()` left the whole
/// suite green — 203 binaries, 5893 tests — and Codecov independently reported
/// ten uncovered lines in `api/validation.rs`.
///
/// `block_omega (ETA_CL, ETA_V) = [1.0, 15.0, 400.0]` is positive definite
/// (`det = 400 − 225 = 175`) and its Cholesky is `L₁₁ = 1`, `L₂₁ = 15`,
/// `L₂₂ = √175 ≈ 13.2`. Only `L₂₁` is out of box: both diagonals pack to
/// `ln(1) = 0` and `ln(13.2) = 2.58`, comfortably inside `±6`, so the single
/// diagnostic is the off-diagonal and the fixture cannot pass by accident.
#[test]
fn an_omega_off_diagonal_above_its_rail_is_reported_as_a_warning() {
    let model = block_omega_model("block_omega (ETA_CL, ETA_V) = [1.0, 15.0, 400.0]");
    let p = &model.default_params;

    // The premise, measured rather than assumed: exactly one coordinate is out
    // of box and it is the off-diagonal.
    let start = pack_with_bounds(p);
    let kinds = coordinate_kinds(p);
    let hits: Vec<_> = coordinates_outside_bounds(&start, &kinds)
        .map(|h| (h.kind, h.side))
        .collect();
    assert_eq!(
        hits,
        vec![(PackedCoordKind::OmegaOffDiagonal, BoxSide::Above)],
        "the fixture must isolate the off-diagonal: {hits:?}"
    );

    let diags = check_packed_start_in_box(p, &FitOptions::default());
    let msg = only_message(&diags);
    assert!(
        !diags[0].is_error(),
        "either rail is a runaway, not a collapse"
    );
    assert_eq!(diags[0].code, "W_INIT_OUTSIDE_BOUNDS");
    assert!(
        msg.contains("Cholesky element"),
        "the off-diagonal arm names the coordinate it clamps: {msg}"
    );
    assert!(
        msg.contains("1.500e1"),
        "the element, at three digits: {msg}"
    );
    assert!(msg.contains("rail of 1e1"), "the rail: {msg}");
    assert!(msg.contains("covariances"), "the remedy: {msg}");

    // The differential half: the same block with a covariance inside the rail
    // is silent, so the assertions above cannot be passing on a blanket
    // rejection of `block_omega`.
    let ok = block_omega_model("block_omega (ETA_CL, ETA_V) = [1.0, 5.0, 400.0]");
    assert!(
        check_packed_start_in_box(&ok.default_params, &FitOptions::default()).is_empty(),
        "L₂₁ = 5 is inside ±10"
    );
}

// ── T14 (#1309 review): a block Ω diagonal is not a declared variance ──────

/// Regression: the Ω-diagonal arm printing `exp(2·packed)` as "a variance".
///
/// On a **diagonal** Ω that is exactly the variance. On a `block_omega` the
/// packed coordinate is `ln(L_ii)`, and `L_ii²` is what is left of the eta's
/// variance once the off-diagonals are accounted for — a number that appears
/// nowhere in the model file. Measured before the fix, on this fixture: the
/// message read *"ETA_V starts at a variance of 2.8888888888888864e5"* while
/// the declaration says `4e5`.
///
/// `block_omega (ETA_CL, ETA_V) = [0.09, 100.0, 4e5]` is positive definite
/// (`det = 36000 − 10000 = 26000`); `L₂₂ = √(4e5 − 100²/0.09) ≈ 537.5`, which
/// packs to `6.29` against the `+6` rail. `L₂₁ = 333` is over its own rail too,
/// so the test asserts on the diagonal entry specifically rather than on
/// `only_message`.
#[test]
fn a_block_omega_diagonal_is_reported_as_a_cholesky_diagonal_not_a_variance() {
    let model = block_omega_model("block_omega (ETA_CL, ETA_V) = [0.09, 100.0, 4e5]");
    let p = &model.default_params;
    let diags = check_packed_start_in_box(p, &FitOptions::default());

    let diag = diags
        .iter()
        .find(|d| d.message.contains("Cholesky diagonal"))
        .unwrap_or_else(|| {
            panic!(
                "a block Ω diagonal must be described as one: {:#?}",
                diags.iter().map(|d| &d.message).collect::<Vec<_>>()
            )
        });
    assert_eq!(diag.code, "W_INIT_OUTSIDE_BOUNDS");
    assert!(diag.message.contains("`block_omega`"), "{}", diag.message);
    // The mechanism, not just the wording: the number quoted is L₂₂², and the
    // message says so rather than calling it ETA_V's variance.
    assert!(
        diag.message.contains("L² = 2.889e5"),
        "quote L², at three digits: {}",
        diag.message
    );
    assert!(
        !diag.message.contains("a variance of"),
        "must not claim the declared variance on a block: {}",
        diag.message
    );
    assert!(
        diag.message.contains("not the variance declared for it"),
        "and must say why: {}",
        diag.message
    );

    // The control: the identical rail hit on a **diagonal** Ω, where `L_ii²`
    // really is the declared variance, keeps the plain wording. Without this
    // arm the assertion above passes for a fix that dropped "a variance of"
    // everywhere.
    let diagonal = box_diags(&params_with_omega("omega ETA_CL ~ 1e8"));
    let dmsg = only_message(&diagonal);
    assert!(
        dmsg.contains("a variance of 1.000e8"),
        "a diagonal Ω keeps the variance wording, and there L_ii² *is* the \
         declared 1e8: {dmsg}"
    );
    assert!(
        !dmsg.contains("Cholesky diagonal"),
        "and does not borrow the block wording: {dmsg}"
    );
    // #1309 review: every computed number is at three digits. `{:e}` printed
    // this rail as `1.6275479141900392e5`.
    assert!(
        dmsg.contains("rail of 1.628e5"),
        "the rail, at three digits: {dmsg}"
    );
}

// ── T18 (#1309 review round 2): an empty box silences one coordinate ───────

/// Regression: `if !diags.is_empty() { return diags; }` — one inverted box
/// suppressing every verdict on every *other* coordinate in the file.
///
/// The comment justifying the early return read "an empty box makes every other
/// verdict on **that coordinate** meaningless", which is true and is narrower
/// than what the code did. `fit()` stops at the first error either way, so the
/// loss was entirely in `ferx check`, whose job is to report a whole model in
/// one pass: a file with a swapped `TVCL` bound *and* an `omega ~ 1e8` reported
/// only the first.
///
/// Asserted as a **pair**: the same `omega ~ 1e8` with and without the inverted
/// θ beside it. Without the second half a fix that simply stopped reporting the
/// inverted box would pass.
#[test]
fn an_empty_box_silences_its_own_coordinate_and_no_other() {
    let both = box_diags(
        "  theta TVCL(1.0, 5.0, 2.0)\n\
         \x20 theta TVV(50.0, 0.1, 500.0)\n\
         \x20 omega ETA_CL ~ 1e8\n\
         \x20 omega ETA_V ~ 0.09\n\
         \x20 sigma PROP_ERR ~ 0.04 (sd)\n",
    );
    let codes: Vec<&str> = both.iter().map(|d| d.code.as_str()).collect();
    assert_eq!(
        codes,
        ["E_INIT_BOUNDS_INVERTED", "W_INIT_OUTSIDE_BOUNDS"],
        "both coordinates must be reported in one pass: {both:#?}"
    );
    assert!(both[0].message.contains("TVCL"), "{:?}", both[0].message);
    assert!(both[1].message.contains("ETA_CL"), "{:?}", both[1].message);
    // The inverted coordinate itself gets exactly one verdict — `packed < lower`
    // and `packed > upper` are both true on an empty box, so the second walk
    // must not pile on.
    assert!(
        !both[1].message.contains("TVCL"),
        "the inverted coordinate is claimed once: {:?}",
        both[1].message
    );

    // The control: the same Ω on a model whose θ is fine. Identical verdict, so
    // the pair above is measuring the suppression and not the Ω arm.
    let omega_only = box_diags(
        "  theta TVCL(5.0, 0.1, 10.0)\n\
         \x20 theta TVV(50.0, 0.1, 500.0)\n\
         \x20 omega ETA_CL ~ 1e8\n\
         \x20 omega ETA_V ~ 0.09\n\
         \x20 sigma PROP_ERR ~ 0.04 (sd)\n",
    );
    assert_eq!(omega_only.len(), 1, "{omega_only:#?}");
    assert_eq!(omega_only[0].message, both[1].message);
}

/// A θ whose box is empty is claimed by the empty-box arm **only** — the
/// declared-range walk must not report it a second time.
///
/// `theta TVCL(1.0, 5.0, 2.0)` has `1.0 < 5.0`, so
/// `theta_outside_declared_range` fires on it in isolation; what keeps the user
/// from seeing two contradictory messages about one declaration is the
/// `inverted` consult, and this is the assertion that consult can fail.
#[test]
fn an_inverted_theta_is_not_also_reported_against_its_declared_range() {
    let params = params_with_theta("theta TVCL(1.0, 5.0, 2.0)");
    let model = model_with_parameters(&params);
    let p = &model.default_params;

    // Premise: the declared-range walk *does* claim this coordinate on its own.
    assert_eq!(
        crate::estimation::parameterization::theta_outside_declared_range(p).count(),
        1,
        "premise: 1.0 is below the declared lower of 5.0, so the walk fires \
         and the `inverted` consult is what suppresses it"
    );
    let diags = box_diags(&params);
    assert_eq!(diags.len(), 1, "{diags:#?}");
    assert_eq!(diags[0].code, "E_INIT_BOUNDS_INVERTED");
}

// ── T19 (#1309 review round 2): the clamp's precondition ───────────────────

/// Regression: `clamp_to_bounds` silently accepting an empty box in debug.
///
/// `f64::clamp` panics on `min > max` deep inside `core::num`, with a message
/// that names neither the coordinate nor the guarantor. Every call site is
/// downstream of `fit_inner` today, so the panic is unreachable — but that is a
/// property of the call graph, not of the function, and a future producer of a
/// *narrowed* box reopens it. The `debug_assert!` names
/// `check_packed_start_in_box` so such a caller learns where the invariant
/// comes from.
///
/// Tests run in debug, so the assertion is live here.
#[test]
#[should_panic(expected = "E_INIT_BOUNDS_INVERTED")]
fn clamping_into_an_empty_box_names_its_guarantor() {
    let bounds = crate::estimation::parameterization::PackedBounds {
        lower: vec![5.0],
        upper: vec![2.0],
    };
    let mut x = [3.0];
    crate::estimation::parameterization::clamp_to_bounds(&mut x, &bounds);
}
