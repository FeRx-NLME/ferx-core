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
    for line in [
        "sigma PROP_ERR ~ 0.0 (sd)", // packs to ln(1e-10) = -23.03, below -8
        "sigma PROP_ERR ~ 1e6 (sd)", // packs to +13.8, above +5
    ] {
        let diags = box_diags(&params_with_sigma(line));
        let msg = only_message(&diags);
        assert!(
            !diags[0].is_error(),
            "{line}: Σ recovers from its rails — {msg}"
        );
        assert_eq!(diags[0].code, "W_INIT_OUTSIDE_BOUNDS");
        assert!(msg.contains("PROP_ERR"), "{msg}");
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
/// Asserted as **sets of `(index, side)`** over a template carrying an
/// out-of-box coordinate of every reachable kind and side at once — not as
/// `diags.len() == 1` on a single-coordinate fixture, which stays green when a
/// kind moves from one consumer to the other.
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
    assert_eq!(
        all.len(),
        4,
        "fixture must carry every reachable kind and side: {all:?}"
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
/// The ten entries below are all `OmegaDiagonal` / `Below`, i.e. all already in
/// #1229's scope: eight simulate-only adaptive-dosing models declaring
/// `omega ~ 1e-10`, and two `reset_init_snapshot_*` anchors declaring `~ 0.0`.
/// **Zero** θ, Σ, off-diagonal or upper-rail hits — which is what makes this
/// change safe for everything shipped.
#[test]
fn no_shipped_model_file_starts_outside_its_box_except_the_known_ten() {
    let allowed: std::collections::BTreeSet<&str> = [
        "adaptive_platelet_ladder.ferx::ETA_CL",
        "adaptive_vanco_auc.ferx::ETA_CL",
        "adaptive_vanco_iov.ferx::ETA_CL",
        "adaptive_vanco_iov_loading.ferx::ETA_CL",
        "adaptive_vanco_loading.ferx::ETA_CL",
        "adaptive_vanco_renal.ferx::ETA_CL",
        "adaptive_vanco_renal_iov.ferx::ETA_CL",
        "adaptive_vanco_renal_loading.ferx::ETA_CL",
        "reset_init_snapshot_fit.ferx::ETA_CL",
        "reset_init_snapshot_occ.ferx::ETA_CL",
    ]
    .into_iter()
    .collect();

    let mut found = std::collections::BTreeSet::new();
    let mut parsed_files = 0usize;
    for dir in ["examples", "nonmem_anchor"] {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.extension().and_then(|s| s.to_str()) != Some("ferx") {
                continue;
            }
            let Ok(src) = std::fs::read_to_string(&path) else {
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
            let file = path.file_name().unwrap().to_string_lossy().to_string();
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
        }
    }

    // Guard against the walk finding nothing because the paths moved: this is a
    // positive count, so an empty sweep fails rather than passing vacuously.
    assert!(
        parsed_files >= 100,
        "expected the shipped model corpus, parsed only {parsed_files} files — \
         has the working directory or the layout changed?"
    );
    let found_refs: std::collections::BTreeSet<&str> = found.iter().map(String::as_str).collect();
    assert_eq!(
        found_refs, allowed,
        "the set of shipped out-of-box coordinates changed"
    );
}
