//! Positive proof that the `debug_assert!` guards in this crate are LIVE (#344).
//!
//! Sibling test file rather than an inline `#[cfg(test)] mod`, for the ordinary
//! reason (`CLAUDE.md`'s sibling-`*_tests.rs` pattern) and for one specific to what
//! this module does: the interior of a `debug_assert!` is, by construction,
//! unreachable under a release-derived profile. Inline in `src/lib.rs` those two
//! lines are permanently-missed patch lines in every Codecov job — the exact defect
//! filed as #1248 — and they took this PR's own patch gate to 9/11 = 81.81%. No
//! test can cover them; they are a measurement artefact, not a coverage gap. Here
//! they are simply not measured, which is how the other 53 `*_tests.rs` files in
//! this repo are already treated.

/// The env var `tools/preflight.sh`'s `debug-assertions` group sets. Deliberately
/// opt-*in*: every other job legitimately runs with the guards off, so an
/// unconditional assertion here would fail `Tests + coverage (core)` and both
/// coverage jobs for doing exactly what they are supposed to do.
const DEMAND: &str = "FERX_REQUIRE_DEBUG_ASSERTIONS";

#[test]
fn debug_assert_guards_run_when_the_gate_demands_them() {
    let demanded = std::env::var_os(DEMAND).is_some();

    // Observe the MACRO, not `cfg!(debug_assertions)`. The property the job exists
    // to establish is "the condition of a `debug_assert!` is evaluated"; reading the
    // cfg flag is one indirection away from it.
    //
    // `Cell` rather than `let mut`: with the guards off the assignment is compiled
    // away, and a `mut` binding that is then never mutated warns.
    let evaluated = std::cell::Cell::new(false);
    debug_assert!({
        evaluated.set(true);
        true
    });
    let evaluated = evaluated.get();

    assert_eq!(
        evaluated,
        cfg!(debug_assertions),
        "`debug_assert!` and `cfg!(debug_assertions)` disagree, which should be \
         impossible — the macro is defined in terms of that cfg. Read this as the \
         canary itself being broken rather than the profile being wrong."
    );

    // The gate. Note the shape: `!demanded || evaluated`, so the only run this can
    // fail is one that asked for the guards and did not get them.
    assert!(
        !demanded || evaluated,
        "{DEMAND} is set — this run came from `tools/preflight.sh debug-assertions` \
         or the `Tests (debug-assertions)` CI job — but `debug_assert!` compiled to \
         nothing, so every guard in the crate is dead and the job is green for no \
         reason. Something switched debug-assertions off for the dev profile: a \
         `[profile.dev] debug-assertions = false` in Cargo.toml, or a \
         `CARGO_PROFILE_DEV_DEBUG_ASSERTIONS` in the environment (#344)."
    );
}
