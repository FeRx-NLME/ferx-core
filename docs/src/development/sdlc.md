# Development Lifecycle (SDLC)

This page describes how the FeRx project is developed across its two coupled
repositories — **ferx-core** (the Rust NLME engine) and **ferx-r** (the R
wrapper package) — from idea to release. It is the canonical process document
for contributors and AI agents working on either repo.

FeRx follows an **iterative, trunk-based continuous-delivery** model (close to
Agile): small changes flow continuously to `main`, releases are tags, and the
loop repeats. The scarcest planning/feasibility input is often not engineering
time but **AI token budget** — treat it as first-class when scoping and
sequencing work (a heavy multi-agent review or a large refactor is a *cost* to
plan for, see [§4](#4-the-development-workflow-with-ai)).

> **Scope.** This covers governance, the issue→PR→merge→release flow, the
> git model, cross-repo synchronization, the testing pyramid, NONMEM
> validation, and the quality gates enforced by CI. It complements (does not
> replace) each repo's `CLAUDE.md`, which carries the low-level
> build/test/style rules.
>
> **Want to help?** You don't need to be on the core team — see
> [Contributing to FeRx development](contributing.md).

---

## 1. Philosophy and quality bar

FeRx is a numerical estimation engine. A wrong answer that *looks* right is
worse than a crash. Because the project is openly AI-assisted, it is
(fairly or not) perceived as an "AI-generated" product, which raises — not
lowers — the bar we hold ourselves to.

**Principles**

- **High quality bar, especially for core numerics.** Aim *higher* than
  comparable tools on testing and validation. Every numerical result must be
  reproducible and validated (see [§7](#7-gold-standard-validation)).
- **Match dev speed to risk.** Fast iteration is welcome for simple,
  non-core features and fixes. **Deliberately slow down** for anything that
  touches estimation, likelihood, PK/ODE math, or the `.ferx`/DSL surface.
- **Parity first, novelty second.** The initial priority is parity with
  **NONMEM**, then Monolix / nlmixr2. Novel features (copula-IIV, neural ODEs,
  FREM, …) come *after* the core is trusted. The emphasis will shift toward
  novel features as the foundation matures.
- **Avoid FeRx becoming a "Christmas tree"** Not every request belongs in FeRx. Evaluate
  each issue/PR against the design philosophy and **prioritize the majority of users**.
  It is fine — expected — to push back, defer, or decline, even for PRs from well-known developers. 
  Deferral is a valid outcome if feature is not a priority: log it as a backlog issue.

**Gate questions for any incoming issue or PR**

1. Does it fit the design philosophy, or does it bloat the tool?
2. Does it serve the majority of users, or a niche of one?
3. For a PR: is the code well-designed, well-documented, and well-tested?
4. Can it be deferred to a future cycle without harm?

---

## 2. Team and governance

| Role | Members | Responsibility |
|------|----------------------|----------------|
| **Core team** (≤ ~5) | TBD | Direction, design authority, merge rights, releases |
| **Power users / developers** (10+) | TBD |Feature contributions, domain review |
| **Advisory board** | TBD | Scientific guidance, validation perspective |
| **Pharma partners** | — | Real-world use cases, guidance |

**Governance guardrails**

- **No single controlling entity.** Explicitly avoid one organization controlling the project's direction.
- **Avoid a bloated organization.** Keep the decision-making core small and the process lightweight.
- Design-level decisions (DSL changes, estimation defaults, breaking API
  changes) require **core-team** sign-off.

---

## 3. Communication

- **Google Chat** for day-to-day coordination.
- **(Bi-)weekly meetings** for planning and design discussion.
- **Technical discussion happens on GitHub** — in the relevant Issue or PR —
  so it is visible, searchable, and tied to the code. Decisions made in chat
  or meetings that affect design should be written back into the issue/PR.

---

## 4. The development workflow (with AI)

FeRx is developed with Claude Code (CC) as a primary implementer, with humans
owning intent, design, and final judgment. The standard loop:

1. **Open a Backlog Issue** in the GitHub Project view. Describe the intent
   clearly — **Why, How, and the design choices**. This gives visibility to
   the rest of the team and forces the problem to be laid out before code is
   written.
2. **Review the issue and agree the design before coding.** Have CC rewrite the
   issue for clarity, then settle the *approach*. For anything touching
   estimation, the numerics, or the `.ferx`/DSL surface, this is an explicit
   **design gate**: write the chosen approach and the alternatives weighed into
   the Issue (or a short design note) and get **core-team buy-in first** — don't
   discover the design in the diff. Simple, non-core changes can skip straight
   to implementation.
3. **Have CC implement the issue and open the PR.** Fill every section of the
   [PR template](#6-pull-requests) — including the cross-repo dependency table.
4. **For large / complex PRs — especially with numerical consequences — run
   the multi-agent review** (`/code-review` / "ultrareview", at medium or high 
   effort with up to ~9 agents),
   then have CC fix what it finds. This is token-expensive: if the budget is
   tight, schedule it overnight or hand it to someone with budget.
5. **Produce a NONMEM comparison** of the fit (see
   [§7](#7-gold-standard-validation)). Wire it into the test suite at the right tier:
   a real convergence fit → **slow tests**; a simulation or quick numeric
   check → **regular tests**.
6. *(Optional, for complex issues)* have **Copilot** review afterward as a
   second independent pass.
7. **Large / complex PRs, or anything affecting design or the DSL, always get
   a human review** in addition to AI review.
8. **Log any remaining or deferred work as new Issues** — nothing important
   should live only in a PR comment.
9. **Manually test an example fit**, ideally on real data and/or a model not
   already used by CC. If it misbehaves, have CC fix it.
10. **Merge.** Before merging, confirm the **Rust and R sides are in sync**,
    documentation is updated, and the feature/fix is fully tested.

> **Rule of thumb on AI autonomy:** let CC move fast on mechanical and
> non-core work; insert human checkpoints (design discussion, manual fit,
> human review) precisely where a subtle numerical error would be expensive.

---

## 5. Git workflow

### 5.1 Branching model — trunk-based

**`main` is always the latest code.**

- New feature/fix work is **branched off `main`**.
- When done (reviewed, tested, in sync), the branch is **merged into `main`**.
- A **release is a tag on `main`** (e.g. `v0.3.0`); see [§9](#9-releases).

```
main ──●──●──●──●──●──▶   tag v0.3.0
        \   /  \   /
 feat/x  ●─●    \ /
 fix/y         ●──●
```

**Trade-off we accept:** `main` is always newest but is *not guaranteed
release-stable* between tags. We mitigate this with the CI gates in
[§8](#8-cicd-and-quality-gates) and by treating tagged releases — not arbitrary
`main` commits — as what we point users at.

> This is intentionally the simplest model. If instability on `main` becomes a
> recurring problem, the alternative (`dev` + release branches) can be revisited
> by the core team — but the default is trunk-based.

**Hygiene**

- **Always pull the latest `main` before starting new work**, and branch off it.
- Keep branches short-lived and focused on one issue.
- Branch/PR titles follow `type(scope): short description [closes #N]` — see
  the PR template header for the allowed `type`/`scope` values.

### 5.2 Experimental / large features

Big experimental features (e.g. copula-IIV, neural ODEs, FREM) are the
exception to "merge small, merge often":

- Develop them on a **long-lived feature branch** until production-ready —
  **do not** merge them into `main` piece-by-piece in a half-working state.
- They must be **documented with multiple examples**.
- They must be **referenced to the literature** (or carry a vignette, if the
  method is genuinely new).

### 5.3 Git worktrees

By default git checks out one revision per folder, so you work one issue at a
time. **Worktrees** let you check out multiple branches at once, so CC can work
several issues in parallel. The cost: local testing is harder (e.g. for the R
package you must install from the worktree folder).

- **Avoid worktrees** unless you have **two or more complex issues** that can't
  each be finished quickly.
- **Always create the worktree off `main`.**
- **In ferx-r specifically**, use `EnterWorktree` at the start of any session on
  a non-`main` branch — it stops uncommitted WIP from one chat contaminating
  another chat that shares the same checkout directory.

---

## 6. Pull requests

Every PR uses `.github/PULL_REQUEST_TEMPLATE.md` and must fill **every**
section before `gh pr create`. Key sections that exist specifically because of
how FeRx is built:

- **Cross-repo dependency table** — record the matching ferx-r (or ferx-core)
  PR and whether it must merge *before / after / together*. If a sibling PR is
  required but not yet open, mark the PR **Draft**. See [§10](#10-cross-repo-synchronization).
- **Numerical validation** — reference values vs NONMEM / nlmixr2 / a prior
  ferx commit (see [§7](#7-gold-standard-validation)).
- **Breaking changes** — `.ferx` format, `FitResult`/sdtab fields, or public
  Rust API. sdtab/`FitResult` changes are parsed by ferx-r → link the R PR.
- **Tests**, **Docs**, **Example**, and the **example-execution checklist**
  (build ferx-core, rebuild ferx-r against it, run affected examples).

**R-side pre-PR gates (ferx-r)**

- Run `roxygen2::roxygenize()` and **commit the regenerated `man/*.Rd`** — they
  are checked into the repo and must stay in sync with the `#'` comments in
  `R/`. Never hand-edit the auto-generated `R/extendr-wrappers.R`.
- **No non-ASCII characters in `R/*.R`** — `R CMD check --as-cran` requires pure
  ASCII (use `-` / `...`, not `—` / `…`). Verify before every PR:

  ```bash
  python3 -c "import glob; [print(f) for f in glob.glob('R/*.R') if any(b>127 for b in open(f,'rb').read())]"
  ```

**Review requirements**

- All PRs: green CI + at least the AI review pass.
- Large/complex, numerical, design-, or DSL-affecting PRs: **a human core-team
  review is mandatory** on top of AI review.

---

## 7. Gold-standard validation

**Every feature that produces numerical results requires a comparison with
NONMEM output**, and every new feature requires a test (lowest tier that fits).

- NONMEM comparison is **essential for core features**. nlmixr2 parity is a
  fine *first pass*, but **NONMEM parity is what matters** — unless it's clear
  nlmixr2 is the better reference.
- CI has **no NONMEM access**, so commit the **NONMEM output files** alongside
  the test/docs rather than running NONMEM in CI.
- Reference NONMEM 7.5+ runs are reproducible in the `pmx` container.

Put the comparison either in the feature's docs page (`docs/src/...`,
`docs/src/faq.md`, or the relevant `estimation/*.md`) or in the PR description.

---

## 8. CI/CD and quality gates

### 8.1 Testing pyramid

Tests live in three tiers (see `CLAUDE.md` for placement rules). Put a new
test in the **lowest tier whose constraints it fits**.

| Tier | What | Where | When it runs |
|------|------|-------|--------------|
| **1 — Unit** | Smallest helper isolating the behaviour; core functionality, math/predictions, fast numeric checks vs NONMEM (e.g. IPRED for ADVAN1-5). Avoid `fit()`. | inline `#[cfg(test)]` in `src/**` | Every PR. **Whole suite must stay < 10 min** (target seconds). |
| **2 — Integration** | Public API (`fit()`, `predict()`, …) but returns immediately — `Ok` after a few outer iterations, or an `Err`. **No convergence loops.** | `tests/*.rs` | Compile-checked every PR; run nightly. |
| **3 — Slow / convergence** | Full population fits to convergence; NONMEM / nlmixr2 fit comparisons. | `tests/*.rs` or `src/`, gated behind `slow-tests` | Nightly (`slow-tests.yml`, 06:00 UTC) + on push to `main` touching estimation code. |

**Real-data tests** are currently **manual** (a handful of models Ron re-runs
after each major feature) and not yet on GitHub — a known gap to formalize.

### 8.2 CI workflows (ferx-core)

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `ci.yml` | PR + push to `main` | `check`, `test` (`--lib --release`), `survival` (TTE feature), `clippy`, `fmt`; **`coverage`** on weekly schedule / manual |
| `slow-tests.yml` | nightly 06:00 UTC, manual, push to `main` touching estimation paths | Tier-3 fits to convergence (`--features ci,slow-tests --release`, `--no-fail-fast`) |
| `docs.yml` | push to `main` touching `docs/**` | `mdbook build` → deploy to `gh-pages` |
| `release.yml` | tag `v*` or manual | GitHub release with generated notes |

> Note: `RAYON_NUM_THREADS=1` is pinned in CI so cargo owns the test
> parallelism on 2-core runners. Don't remove it.

### 8.3 CI workflows (ferx-r)

`R-CMD-check.yaml` and `test-coverage.yaml`. **ferx-r CI builds ferx-core from
the commit pinned in `ferx-r/src/rust/Cargo.lock`** — not from latest `main`.
See [§10](#10-cross-repo-synchronization).

### 8.4 Code-quality thresholds

| Tool | Measures | Target |
|------|----------|--------|
| **Codecov** | unit-test line coverage | **>90%**; a PR must not degrade coverage (Codecov comments on PRs) |
| **CodeFactor** | code quality / smells | **A+** |

`rustfmt` is enforced by the shared pre-commit hook (`git config core.hooksPath
.githooks`) and by the `fmt` CI job; `clippy` must be clean.

### 8.5 Documentation workflow

Docs are an [mdBook](https://rust-lang.github.io/mdBook/) under `docs/`.

- **Edit only `docs/src/`.** `docs/book/` is generated, git-ignored, and
  **never committed** — `docs.yml` builds and deploys it to `gh-pages` on every
  push to `main` touching `docs/**`. Build locally to preview, but leave the
  output untracked.
- **New pages must be linked in `docs/src/SUMMARY.md`** or they won't appear in
  the book.
- Route user-visible changes to the right page: `model-file/fit-options.md`
  (`[fit_options]` keys), `model-file/individual-parameters.md` (DSL syntax),
  `estimation/*.md` (estimator behaviour), `faq.md` (NONMEM / nlmixr2
  comparisons).
- On the R side, the `roxygen2`-generated `man/*.Rd` files **are** the docs —
  regenerate and commit them (see [§6](#6-pull-requests)).

> **To resolve:** the PR template still carries a checkbox "`mdbook build` run
> and `docs/book/` committed," which contradicts the *never commit `docs/book/`*
> rule above. The intent is **not** to commit built output — fix the template
> so the two agree.

### 8.6 Build portability (stock nightly)

ferx-core builds on a **stock nightly** toolchain — no custom compiler. The exact
FOCE/FOCEI and HMC gradients come from hand-rolled analytic `Dual2` sensitivities
(`sens/`), and models outside the provider's scope fall back to finite differences.
The Enzyme autodiff path that used to require a from-source toolchain was retired.

- **Keep the build toolchain-light.** Standard-Rust users (Mac / Linux / Windows)
  and CI install ferx with `cargo build` alone. Do not reintroduce a dependency on
  a custom compiler or a non-default Cargo feature for the core gradient path.

---

## 9. Releases

Trunk-based, so a release is a **tag on `main`**:

1. Confirm `main` is green (CI), ferx-core ↔ ferx-r are in sync, docs are
   current, and the relevant `Cargo.toml` / `DESCRIPTION` versions are bumped.
   *(Current: ferx-core `0.1.6`, ferx-r `0.1.5`.)*
2. Tag `main` with `vX.Y.Z` (e.g. `v0.3.0`). `release.yml` builds the GitHub
   release and generates notes.
3. Point users at the **tagged release**, not an arbitrary `main` commit.
4. For an R-side release, ensure `Cargo.lock` pins the intended ferx-core
   commit (see below) so the build is reproducible.

**Versioning.** Both repos follow **semantic versioning** (`MAJOR.MINOR.PATCH`).
A breaking change to the `.ferx` format, the public Rust API, or sdtab /
`FitResult` fields bumps MAJOR (pre-1.0, bump MINOR); a backward-compatible
feature bumps MINOR; a fix bumps PATCH. The PR template's *Breaking changes*
section is what flags that a MAJOR/MINOR bump is due.

**Changelog.** ferx-core keeps a hand-curated `CHANGELOG.md` (Keep a Changelog
format); add an `[Unreleased]` entry **in the same PR** as any user-facing
change — `CLAUDE.md` carries the rule. ferx-r keeps the equivalent `NEWS.md`, so
a cross-repo change may touch both. `release.yml` additionally auto-generates
GitHub release notes from merged PRs, so keep PR titles in `type(scope): …`
form. User-facing breaking changes also carry **migration notes** (PR template).

**Distribution.** Releases are **GitHub releases** built by `release.yml` on
each `vX.Y.Z` tag. Today ferx-core is consumed as a **git dependency** (not
published to crates.io) and ferx-r is installed from source / GitHub
(`R CMD INSTALL`, e.g. `remotes::install_github`). If we later publish to
crates.io or a CRAN-style channel, add that publish step here.

---

## 10. Cross-repo synchronization

The two repos are coupled, and keeping them in sync is a first-class part of
the lifecycle — **a PR is not "done" until both sides agree.**

**Local builds (automatic).** ferx-r's `src/rust/.cargo/config.toml` carries a
`[patch]` that swaps in `../../../ferx-core` when this checkout exists, so
**local** R builds pick up ferx-core changes with no Cargo edits. Verify a
ferx-core change by rebuilding the R package:

```bash
cargo build --release                       # ferx-core
cd ../ferx-r && R CMD INSTALL .
```

> **Never commit a `Cargo.toml` that flips the ferx-core dep to a path
> dependency.** ferx-r's `src/rust/Cargo.toml` pins ferx-core as a git dep on
> `main`; the `[patch]` already handles the local swap. A committed path dep
> breaks reviewers and CI, which build against GitHub `main`.

**ferx-r CI (manual bump required).** CI does **not** get the change
automatically — it builds from the ferx-core commit pinned in
`ferx-r/src/rust/Cargo.lock`. A ferx-r PR that needs a new ferx-core commit
(e.g. a newly-`pub` API) must bump that lock with:

```bash
ferx-r/tools/update-ferx-core-lock.sh        # never a bare `cargo update`
```

A bare `cargo update` unpins the patch and CI fails with
`error[E0603]: ... is private`.

**Checklist when changing a `pub` API or `FitResult`/sdtab field**

- [ ] Open the matching ferx-r PR and record it in the PR cross-repo table.
- [ ] Bump `Cargo.lock` in ferx-r via `update-ferx-core-lock.sh`.
- [ ] Confirm the R package compiles against the new ferx-core.
- [ ] Update docs on both sides.
- [ ] Coverage not degraded.

---

## 11. Tooling conventions

- **Model:** use **Opus 4.6 or better** for development. **Never Sonnet or
  Haiku** for code (Sonnet is acceptable only for documentation). _(Newer Opus
  releases — 4.7, 4.8 — supersede 4.6 under the same rule.)_
- **Pre-commit hook:** `git config core.hooksPath .githooks` after cloning
  (blocks commits failing `rustfmt`).
- **Sensitivity-reachable code:** keep functions on the `Dual2` path (the `*_g<T:
  PkNum>` closed forms and propagators) differentiable — use `if`/`else` rather than
  `f64::max`/`min` where it matters; see "Analytic Sensitivities" in `CLAUDE.md`.

---

## 12. Definition of done

A change is done only when **all** of the following hold:

- [ ] Implements the issue's stated intent; design choices discussed where open.
- [ ] Tests added at the correct tier; whole fast suite still < 10 min.
- [ ] NONMEM comparison included for any numerical behaviour.
- [ ] CI green: `check`, `test`, `survival`, `clippy`, `fmt`; coverage not degraded.
- [ ] AI review run (and human review for large/numerical/DSL/design PRs).
- [ ] Docs updated (`docs/src/**`, examples) for any user-visible change.
- [ ] ferx-core ↔ ferx-r in sync (R package compiles; `Cargo.lock` bumped if needed).
- [ ] R-side (if touched): `man/` regenerated & committed, `R/*.R` pure ASCII,
      the package still installs with a stock `cargo build`.
- [ ] Deferred/remaining work logged as new Issues.
- [ ] Manually tested on an example fit (ideally real / novel data).

---

## 13. Maintenance & support

Work doesn't stop at a release — most of an estimation engine's life is bug
reports, parity questions, and follow-up fixes.

**Bug triage.** User-reported problems land as GitHub Issues (see
[Contributing](contributing.md)) and are labelled and prioritized in the Project
view. Reproduce first; a confirmed bug gets a **failing regression test before
the fix** (see [§8](#8-cicd-and-quality-gates)). Correctness bugs in the core
numerics outrank cosmetic issues.

**Patch / hotfix releases.** Trunk-based keeps `main` as the newest code, which
is the easy case: an urgent fix branches off `main`, merges, and ships as a
**PATCH tag** (`vX.Y.Z+1`). If `main` already carries unreleased work that
shouldn't go out yet, cherry-pick the fix onto the **latest release tag** on a
short-lived `release/vX.Y` branch and tag the patch from there — this is the
explicit answer to trunk-based's "`main` isn't always release-stable"
trade-off ([§5](#5-git-workflow)). A cross-repo fix must keep ferx-core ↔ ferx-r
in sync ([§10](#10-cross-repo-synchronization)).

**Deprecation & breaking changes.** Avoid silent breaks. A breaking change to
the `.ferx` format, the public API, or sdtab / `FitResult` fields must be
flagged in the PR template's *Breaking changes* section, carry **migration
notes**, bump the version ([§9](#9-releases)), and be announced in the release
notes / `NEWS.md`. Where practical, **deprecate with a warning for one release**
before removing.

**Supported versions & support channel.** We support the **latest release** and
encourage users to upgrade; old release lines aren't maintained unless a
security fix demands it. User-facing support is via **GitHub Issues**; internal
coordination via Google Chat ([§3](#3-communication)).

---

## 14. Security & dependencies

FeRx is a numerical library, not a network service, so its attack surface is
small — but it ships native code (FFI via extendr) and a tree of crate / R
dependencies, all of which need hygiene. We aim for a lightweight **DevSecOps**
posture: push these checks into CI rather than bolt them on later.

**Dependency hygiene.** Keep dependencies current and watch for advisories:

- Run **`cargo audit`** (RUSTSEC advisory DB) over the Rust tree — wired into CI
  as `.github/workflows/audit.yml` (on dependency-file changes, weekly, and on
  demand).
- **Dependabot** (`.github/dependabot.yml`) opens weekly `cargo` and
  `github-actions` update PRs for ferx-core. *(ferx-r: still to add.)*
- Pin the nightly toolchain (`rust-toolchain.toml`) and bump it deliberately;
  a toolchain roll should not silently change numerical behaviour
  ([§8](#8-cicd-and-quality-gates)).

**Native / unsafe code.** Review any `unsafe` and the **extendr FFI boundary**
(`src/rust/src/lib.rs` in ferx-r) with extra care — a memory-safety bug there
can crash the host R session. Prefer safe abstractions; justify any `unsafe` in
the PR.

**Vulnerability reporting.** Report suspected vulnerabilities **privately**, not
in a public Issue — ferx-core's [`SECURITY.md`](https://github.com/FeRx-NLME/ferx-core/security/policy)
documents the GitHub private-advisory flow. *(ferx-r: `SECURITY.md` still to
add.)*

**Security-sensitive changes.** For changes that parse untrusted input, do file
I/O, or touch the FFI boundary, run the **`/security-review`** skill (or request
a focused human review) before merge.
