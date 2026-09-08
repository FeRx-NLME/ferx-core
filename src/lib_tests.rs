//! Positive proof that the `debug_assert!` guards in this crate are LIVE (#344).
//!
//! Sibling test file rather than an inline `#[cfg(test)] mod`, for the ordinary
//! reason (`CLAUDE.md`'s sibling-`*_tests.rs` pattern) and for one that used to be
//! specific to what this module does: the interior of a `debug_assert!` is, by
//! construction, unreachable under a release-derived profile, so inline in
//! `src/lib.rs` those two lines were permanently-missed patch lines in every
//! Codecov job — the defect filed as #1248, which took this file's own PR to
//! 9/11 = 81.81%. #1248 has since been fixed at the source: the coverage jobs build
//! `[profile.ci-cov]`, where the guards are live and their lines are ordinary
//! covered lines. The sibling layout stays for the ordinary reason.

/// The env var that demands the guards be live. Set by `tools/preflight.sh`'s
/// `debug-assertions` group and by the two coverage jobs in `ci.yml`, which since
/// #1248 build `[profile.ci-cov]`. Deliberately opt-*in*: the `Check`/`Clippy` jobs
/// and any `ci-fast` build legitimately run with the guards off, so an
/// unconditional assertion here would fail them for doing exactly what they are
/// supposed to do.
const DEMAND: &str = "FERX_REQUIRE_DEBUG_ASSERTIONS";

/// The mirror image, for the `Tests (release semantics)` lane. That job exists to
/// run the Tier-1 suite the way a *user's* build behaves — guards compiled out, no
/// overflow checks — because #1248 moved both coverage jobs onto `ci-cov` and would
/// otherwise have left no per-PR job exercising release semantics at all (raised by
/// review on PR #1293). Its value is entirely in being the OTHER mode, so it needs
/// the same protection from the opposite direction: if `ci-fast` ever acquired
/// `debug-assertions`, the lane would silently become a duplicate of the coverage
/// jobs, every test would stay green, and the release-only arms of
/// `sim_outcome_category_and_count_continuous_value_is_nan` and friends would go
/// unchecked again — the #344 failure shape with the sign flipped.
const FORBID: &str = "FERX_REQUIRE_NO_DEBUG_ASSERTIONS";

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
         or one of the two coverage jobs in ci.yml — but `debug_assert!` compiled to \
         nothing, so every guard in the crate is dead and the run is green for no \
         reason. Something switched debug-assertions off for the profile being \
         built: a `[profile.ci-cov] debug-assertions = false` in Cargo.toml, a \
         `CARGO_PROFILE_CI_COV_DEBUG_ASSERTIONS` in the environment, or a \
         `--profile` that reverted to `ci-fast` (#344, #1248)."
    );

    // The opposite gate, same shape: `!forbidden || !evaluated`, so the only run
    // this can fail is one that asked for release semantics and got the guards.
    let forbidden = std::env::var_os(FORBID).is_some();
    assert!(
        !forbidden || !evaluated,
        "{FORBID} is set — this run came from `tools/preflight.sh release-semantics` \
         or the `Tests (release semantics)` CI job, whose whole purpose is to exercise \
         the crate the way a user's release build behaves — but `debug_assert!` IS \
         live, so this lane is now a duplicate of the coverage jobs and nothing on \
         the PR checks the guards-off arms. Something switched debug-assertions ON \
         for the profile being built: a `[profile.ci-fast] debug-assertions = true` \
         in Cargo.toml, a `CARGO_PROFILE_CI_FAST_DEBUG_ASSERTIONS` in the \
         environment, or a `--profile` that drifted to `ci-cov` (#1293)."
    );

    // Both set at once is a caller error, not a profile problem: no build can
    // satisfy both, so it would fail whichever way the profile went and the
    // diagnostics above would each blame the manifest.
    assert!(
        !(demanded && forbidden),
        "{DEMAND} and {FORBID} are both set; they are contradictory demands."
    );
}
