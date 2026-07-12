# Plan: Flip-flop absorption robustness follow-ups (#735 / #785 / #786)

**Tracking issues:** [#735](https://github.com/FeRx-NLME/ferx-core/issues/735),
[#785](https://github.com/FeRx-NLME/ferx-core/issues/785),
[#786](https://github.com/FeRx-NLME/ferx-core/issues/786)
**Depends on:** [#790](https://github.com/FeRx-NLME/ferx-core/issues/790) (IG analytical
closed form) — **do this work after #790 lands** (see "The #790 dependency").
**Scope:** ferx-core only (no `pub` API change → no ferx-r follow-up).
**Status:** 📝 Ready to start once #790 merges. This plan is written against the
**post-#790** renamed/generalized machinery.

## TL;DR

Three follow-ups from the Phase-3 flip-flop→ODE-twin reroute (#733) and the #776 adversarial
review. All three live in the twin-less flip-flop guard and the twin-builder desugar — **the
exact code #790 renames from `transit_*` to `absorption_*` and extends to cover IG.** #790
does **not** fix any of them (the guard stays η = 0-only, and stays a panic), but it *widens
every hole to IG as well as transit*. So the right move is: land #790, then fix all three
**generically** (against `absorption_*`), each with a transit **and** an IG test.

Recommended split — two PRs, both after #790:

| PR | Closes | What | Why grouped |
|---|---|---|---|
| **A — guard completeness** | #785 + #786 | Make the twin-less flip-flop guard catch boundary crossings that the η = 0 / point-estimate check misses: a per-subject **EBE** crossing (#785) and a per-**draw** crossing (#786). | Same root cause (guard only samples one parameter point), same two functions (`check_absorption_flip_flop_no_twin`, `absorption_flip_flop_at`), both cheap. |
| **B — twin-builder extension** | #735 | Extend `absorption_ode_equivalent_source` to build an ODE twin for `lagtime=` / `f=` models, so flip-flop transit/IG models carrying lag or bioavailability auto-route instead of returning the closed form's `0`. | Parser-side, separable, larger; current behaviour is already **safe** (loud warning), so lower urgency. |

Priority rationale: **#785 is the only genuinely *silent* one** (a subject's likelihood
degenerates to 0 with no warning) — the exact class #588/#699/#776 set out to kill, so it
leads. #786 is loud (it panics) but aborts an entire uncertainty run, so it rides with #785.
#735 is already loud-and-safe (actionable `W_TRANSIT_FLIP_FLOP`), so it is an ergonomics
upgrade, not a correctness fix — it comes last.

---

## Background: the flip-flop reroute machinery

A transit/IG **closed form** clamps its concentration profile to an identically-zero vector
once the disposition is no faster than absorption — `ke = CL/V ≥ KTR = (n+1)/mtt` for transit,
`ke ≥ 1/(2·MAT·CV²)` for IG (the "flip-flop" regime), or coincident 2-cpt eigenvalues. A zero
profile silently degenerates the objective (a proportional error model collapses `(σ·pred)²`
to 0).

#733/#776 handle this by **rerouting to an ODE twin**: `effective_model_for_eval`
(`src/pk/mod.rs`) swaps the closed form for its numerically-integrated `transit()`/`igd()`
equivalent — valid in *both* regimes — per (θ, η) whenever a twin exists. The twin is built at
parse time by `absorption_ode_equivalent_source` (`src/parser/model_parser.rs`) and stored on
`CompiledModel::absorption_ode_equivalent`.

**The gap the trio addresses:** a **twin-less** model — one carrying a `lagtime=` / `f=`
mapping or a user `[odes]` / `[scaling]` / `[initial_conditions]` block, all of which make the
desugar decline — has nothing to reroute to. #776 added `check_absorption_flip_flop_no_twin`
to reject those, but only samples parameters **at one point** (η = 0 typical values, or the
uncertainty point estimate). The trio are the three ways a boundary crossing slips past that
single-point check:

- **#735** — the twin-builder *could* serve a `lagtime=`/`f=` model but declines, so it stays
  twin-less and gets rejected/returns `0` instead of auto-rerouting.
- **#785** — a model in-domain at η = 0 whose per-subject **EBE** crosses the boundary during
  the inner loop → closed form returns `0` for that subject, **silently**.
- **#786** — a model in-domain at its point estimate whose sampled **draw** crosses the
  boundary → the per-draw `assert_*` **panics**, aborting the whole uncertainty simulation.

---

## The #790 dependency (why: after, not before/concurrent)

#790 (IG closed form) is uncommitted on `worktree-feat+790-ig-analytical-closed-form` and
rewrites all of these files. It **renames the transit-specific machinery to absorption-wide**
and gives IG the same twin + guard. The trio's fix sites all move. Doing the trio *before* or
*concurrent with* #790 would mean fixing `transit_*` functions that #790 deletes → guaranteed
conflicts and duplicated work. **Land #790 first, then target the renamed identifiers.**

Rename map (names are stable; line numbers are pre-landing, re-confirm after #790 merges):

| Old (`main`) | New (post-#790) | File |
|---|---|---|
| `check_transit_flip_flop_no_twin` | `check_absorption_flip_flop_no_twin` | `src/api.rs` |
| `assert_transit_flip_flop_no_twin` | `assert_absorption_flip_flop_no_twin` | `src/api.rs` |
| `transit_flip_flop_at` | `absorption_flip_flop_at` | `src/pk/mod.rs` |
| `transit_ode_equivalent_source` | `absorption_ode_equivalent_source` | `src/parser/model_parser.rs` |
| `CompiledModel::transit_ode_equivalent` | `CompiledModel::absorption_ode_equivalent` | `src/types.rs` |
| `struct TransitOdeEquivalent` | `struct AbsorptionOdeEquivalent` | `src/types.rs` |
| `check_transit_support` | `check_absorption_closed_form_support` | `src/api.rs` |
| `assert_transit_support` | `assert_absorption_closed_form_support` | `src/api.rs` |

**Scope-broadening consequence (important):** post-#790, `absorption_flip_flop_at` has 4 arms
(`OneCptTransit`, `TwoCptTransit`, `OneCptIg`, `TwoCptIg`), and
`absorption_ode_equivalent_source` matches all four PK patterns — but **still declines
`lagtime=`/`f=`** (the allowlists are `{cl,v,n,mtt}` / `{cl,v1,q,v2,n,mtt}` / `{cl,v,mat,cv2}`
/ `{cl,v1,q,v2,mat,cv2}`, no lag/f). So **every hole in the trio now exists for IG too.** Each
fix below must be written against the generalized `absorption_*` code and covered by **both a
transit and an IG test** — a one-time cost that closes the gap for both families at once.

---

## PR A — guard completeness (#785 + #786)

Both bugs are the same defect: `check_absorption_flip_flop_no_twin` /
`assert_absorption_flip_flop_no_twin` only evaluate the flip-flop predicate at **one parameter
point**, but the boundary is parameter-dependent, so a different η (EBE) or a different θ
(draw) can cross it undetected. The building block for the fix already exists —
`absorption_flip_flop_at(model, subject, theta, eta) -> bool` evaluates the predicate at **any**
(θ, η) (it is exactly what `effective_model_for_eval` uses per-eval).

### #785 — EBE-driven flip-flop degenerates a subject silently

**Root cause.** The only reject runs up front at η = 0 typical values (in `fit()` at
`src/api.rs` ~2771 and ~2921 on `main`). A twin-less model that passes there but whose
per-subject EBE drives `ke ≥ KTR` during the inner loop hits `effective_model_for_eval`
(`src/pk/mod.rs`), which — finding no twin — returns the closed form, yielding a zero profile
for that subject. That subject's likelihood contribution silently degenerates. Severity: niche
(twin-less model **and** a subject whose EBE crosses a boundary the typical value didn't), but
**silent** — the worst class.

**Fix (cheap, non-hot-path — a fit-*end* check, per the issue).** `effective_model_for_eval`
returns `&CompiledModel`, not a `Result`, so per-eval plumbing is out. Instead, after
convergence, for a twin-less flip-flop-capable model (`absorption_ode_equivalent.is_none()` and
`pk_model ∈ {OneCptTransit, TwoCptTransit, OneCptIg, TwoCptIg}`), loop subjects and evaluate
`absorption_flip_flop_at(model, subject, &final_theta, &final_ebe)` at each subject's **final
EBE**. If any crossed, push a `W_ABSORPTION_FLIP_FLOP_EBE` warning naming the subject(s) and
pointing to the ODE `transit()` / `igd()` forcing form.

- **Hook point.** Where `fit()` finalizes warnings after EBEs are known — near
  `result.warnings = accumulated_warnings;` (`src/api.rs` ~4991 on `main`) or the later
  `fit_result.warnings.push` sites (~5613). The final per-subject EBEs are computed for sdtab
  output; hook the check alongside that so no EBE is re-solved. **Open item:** confirm where the
  final EBE array is materialised post-#790 (FitResult exposes `ebe_kappas` but the η EBEs flow
  through the sdtab/diagnostics path — locate and reuse them, do not recompute).
- **Warning code.** Add `W_ABSORPTION_FLIP_FLOP_EBE` to the typed `WarningCode` registry
  (the #781/#791 at-source typed-warning machinery) — category `Absorption` if one exists, else
  the closest fit.

**Tests (Tier 1/2, no NONMEM — pure diagnostics):**
- A twin-less transit model (e.g. `pk one_cpt_transit(..., lagtime=LAG)`) in-domain at η = 0 but
  with a fixture subject whose EBE crosses `ke ≥ KTR` → `fit()` returns `Ok` **with** the
  `W_ABSORPTION_FLIP_FLOP_EBE` warning naming that subject.
- The IG twin (`pk one_cpt_ig(..., lagtime=LAG)`) analogue → same warning fires.
- A twin-*carrying* model with the same EBE crossing → **no** warning (it reroutes per-eval).
- Negative: an in-domain twin-less model at all EBEs → no warning.

### #786 — `simulate_with_uncertainty` panics, aborting all draws

**Root cause.** `simulate_with_uncertainty` (`src/api.rs` ~9218 on `main`) loops draws and
calls `simulate_inner_with_draw` per draw; that shared chokepoint calls the **panicking**
`assert_absorption_flip_flop_no_twin(model, population, &params.theta)` on the *draw's* θ (~8112
on `main`, ~7679 post-#790). A point-estimate-in-domain model with a draw that lands in
flip-flop **panics the entire run** — even though `simulate_with_uncertainty` is
`Result`-returning and already has per-subject degenerate-draw handling (censor + warn, #762/#763).

**Fix (well-scoped, per the issue).** In `simulate_with_uncertainty`'s per-draw loop, before
calling `simulate_inner_with_draw`, pre-check the draw with the **`Option`-returning**
`check_absorption_flip_flop_no_twin(model, population, &params.theta)`. If `Some(msg)`: push it
into the existing `sim_warnings` buffer (already threaded — `src/api.rs` ~9263) and `continue`
(skip that draw) so the other draws still produce a result. Because the bad draw never reaches
`simulate_inner_with_draw`, its `assert_*` at ~7679 never fires. **Leave the `assert_*` on the
single-shot `predict()` / `simulate()` / `simulate_inner_with_draw` paths unchanged** (the issue
is explicit) — those genuinely should fail loudly.

- Decide skip-and-warn (preferred) vs. whole-sim `Err`. The issue prefers **skip-and-warn**;
  it matches the #762/#763 "run continues, degenerate cases named" precedent on this exact path.
- Note: `simulate_with_uncertainty`'s early point-estimate guard (via `simulate_with_options`
  ~7320 post-#790) stays — this only changes the *per-draw* handling.

**Tests (Tier 2, no NONMEM):**
- A twin-less transit model whose point estimate is in-domain but with a seeded covariance that
  yields ≥1 flip-flop draw → `simulate_with_uncertainty` returns `Ok`, the surviving draws are
  present, and a warning names the skipped draw(s). (Was: panic.)
- IG analogue.
- Negative: all draws in-domain → no warning, full draw count returned.
- Unchanged-contract guard: single-shot `simulate()` on a twin-less flip-flop model still
  panics (assert path untouched).

---

## PR B — twin-builder extension for `lagtime=` / `f=` (#735)

**Root cause.** `absorption_ode_equivalent_source` (`src/parser/model_parser.rs` ~6623
post-#790) returns `None` — declining to build a twin — the moment any `pk(...)` role key falls
outside the disposition allowlist (the `roles.keys().any(|k| !allowed.contains(...))` gate). A
`lagtime=`/`f=` mapping trips it, so a flip-flop `pk one_cpt_transit(cl,v,n,mtt, lagtime=LAG)`
(and its 2-cpt / IG cousins) has no twin and keeps returning the closed form's `0`.

**Current behaviour is safe, not silent:** those cases emit the actionable
`W_TRANSIT_FLIP_FLOP` ("rewrite it as an explicit ODE transit model") and are caught by
`transit_flip_flop_without_twin_warns_actionable` in `tests/transit_analytic_equivalence.rs`.
So this is an **ergonomics upgrade** (auto-route instead of manual rewrite), not a correctness
fix — hence last.

**Fix.** Extend the desugar to build a twin for `lagtime=`/`f=` models across **all four**
patterns (1-/2-cpt × transit/IG):
1. Add `lagtime` / `f` to the accepted keys (they must not trip the "outside scope" return).
2. Emit them onto the generated ODE model so the ODE `transit()` / `igd()` forcing applies them
   — F scales the dose mass into the input-rate compartment, lagtime shifts `tad`. The ODE
   forcing path already supports both (indexed-F #486/#496, lag #472).
3. Keep user `[odes]` / `[scaling]` / `[initial_conditions]` **out of scope** — no unique
   desugar (the issue is explicit); those stay twin-less and rely on PR A's guard completeness.

**Open design question (resolve before coding):** exactly *how* to thread `lagtime`/`f` into the
generated `.ferx` source. The closed form binds them via the `pk(...)` role map; the ODE twin
uses `ode(...)` + `[odes]`. ODE models bind F/lag via either a `pk(f=F, lagtime=LAG)` mapping or
bare `f=`/`lagtime=` individual parameters routed to `RESERVED_PK_SLOTS`
(`src/parser/model_parser.rs` ~2786). Determine which form the generated source should emit so
the ODE `transit()`/`igd()` path reads F/lag correctly, and pin it with the equivalence test
below. (This is the one genuinely new mechanism in the trio.)

**Tests:**
- **Equivalence (the internal anchor).** Extend `tests/transit_analytic_equivalence.rs` (and the
  new `tests/ig_analytic_equivalence.rs` from #790): a flip-flop transit/IG model *with* `lagtime`
  and with `f` now auto-routes to its twin and matches a hand-written ODE `transit()`/`igd()`
  reference carrying the same lag/F — to closed-form↔ODE tolerance. Replace the "warns actionable"
  assertion for the now-supported lag/f cases with an "auto-routes + matches twin" assertion.
- **NONMEM anchor (CLAUDE.md mandate — this changes numeric output).** A flip-flop
  transit-with-lag model fit/predicted vs an equivalent NONMEM `$DES` transit run (a previously-
  zero-returning model now produces a real profile). Slow-tests-gated, in the example page or PR
  description. The analytic≡ODE equivalence test is the fast per-PR backstop; the NONMEM run is
  the external validation.
- Negative: a user-`[odes]` flip-flop model still declines the twin and still warns actionably
  (unchanged — out of scope).

---

## Sequencing & effort

1. **#790 merges** (prerequisite — do not start before).
2. **PR A (#785 + #786)** — guard completeness. Small; pure diagnostics/robustness; no NONMEM.
   Lands the silent-degeneration fix (#785) and the crash fix (#786) together. **Highest
   correctness value.**
3. **PR B (#735)** — twin-builder extension. Medium; one design decision + a NONMEM anchor +
   equivalence tests. Ergonomics upgrade over already-safe behaviour.

Each PR: `[Unreleased]` CHANGELOG entry (Fixed for A, Added/Changed for B), the per-diff Codecov
patch ≥ 90% (every new guard branch / desugar branch needs its negative-or-edge test), and a docs
touch if user-visible (B changes which models auto-route — note it in `docs/model-file/absorption.qmd`
and/or `docs/warnings.qmd`; A adds a warning code → `docs/warnings.qmd`).

## Open questions / items to confirm post-#790

- **[#785]** Where the final per-subject η EBEs are materialised at fit-end post-#790 (reuse the
  sdtab path's array; don't recompute). Confirm the `WarningCode` category for
  `W_ABSORPTION_FLIP_FLOP_EBE`.
- **[#786]** Confirm the per-draw loop still funnels through `simulate_inner_with_draw` post-#790
  (agent saw it at ~7679) and that `sim_warnings` is still threaded (~9263 on `main`).
- **[#735]** The emission form for `lagtime`/`f` in the generated ODE-twin source (pk-map vs bare
  reserved-slot individual parameters) — the one blocking design decision.
- **[all]** Re-confirm the rename table's line numbers against #790 as merged (names are stable;
  numbers will have drifted from the pre-landing worktree).

## Verification

- `cargo check --no-default-features --features ci`; push for the matrix.
- `cargo test --lib` + the extended `tests/{transit,ig}_analytic_equivalence.rs` (Tier 2).
- `cargo +nightly llvm-cov --tests --no-default-features --features ci` → patch ≥ 90% on each diff.
- PR B: run the slow-tests-gated NONMEM anchor locally (`--features slow-tests`) and record the
  OFV/estimate comparison in the PR.
