# Contributing to FeRx development

You don't need to be on the core team to move FeRx forward. There are three
ways to contribute, in increasing order of involvement — pick whichever fits
your time and expertise. **All three start or end as a GitHub Issue or PR**, so
the work stays visible to everyone (see [Communication](sdlc.md#3-communication)).

For the full process behind any of this, see the
[Development Lifecycle (SDLC)](sdlc.md).

## Test it and tell us what breaks

The single most valuable thing most users can do is **run FeRx on real models
and data and report where it disagrees** with NONMEM / Monolix / nlmixr2 — or
where it is slow, unclear, or crashes. Parity with established tools is our top
priority (see [Gold-standard validation](sdlc.md#7-gold-standard-validation)),
so a concrete, reproducible mismatch is gold.

- Install per the [Installation](../installation.md) page, then run your model
  via the CLI or the R package (`ferx_fit()`).
- Compare the key outputs — **OFV, theta, omega, sigma, ETAs, CWRES/IWRES** —
  against your reference tool.
- If something is off, capture the two output sets and the versions, and open
  an Issue (next section). Even "it ran but the OFV is 30 points off NONMEM" is
  a useful report.

## Open an Issue

Issues are where all work starts (see [the development
workflow](sdlc.md#4-the-development-workflow-with-ai)): bug reports, parity
gaps, feature requests, and docs problems all belong here.

How to write a good one:

- **Search existing issues first** to avoid duplicates.
- **Bug:** a minimal `.ferx` model + a small data subset that reproduces it,
  what you expected vs. what you got, and the ferx-core / ferx-r versions.
- **Parity gap:** attach the NONMEM (or other tool) output and the ferx output
  side by side, plus the model and data.
- **Feature:** state **Why, How, and the design choices**, and who benefits.
  Expect push-back or deferral if it serves a niche of one — see the
  "Christmas tree" guardrail in [Philosophy and quality
  bar](sdlc.md#1-philosophy-and-quality-bar).
- Add it to the GitHub **Project view** so the team has visibility and can
  triage it.

You don't have to fix what you file — a well-described Issue is a complete
contribution on its own.

## Open a Pull Request

Ready to write code or docs? The full process is in the
[SDLC](sdlc.md#4-the-development-workflow-with-ai); the short version for a
contributor:

1. **Claim the work.** Comment on the Issue you want to take (or open one
   first) so effort isn't duplicated.
2. **Branch off the latest `main`** (trunk-based — see [Git
   workflow](sdlc.md#5-git-workflow)); keep it focused on one issue.
3. **Test it.** Add a test at the right tier (see the [testing
   pyramid](sdlc.md#8-cicd-and-quality-gates)); for anything numerical add a
   NONMEM comparison ([Gold-standard
   validation](sdlc.md#7-gold-standard-validation)). Every feature needs a
   test; every bug fix needs a regression test that fails without it.
4. **Keep both sides in sync.** ferx-core ↔ ferx-r ([Cross-repo
   synchronization](sdlc.md#10-cross-repo-synchronization)) and docs updated;
   on the R side, regenerate `man/` and keep `R/*.R` ASCII-only (see [Pull
   requests](sdlc.md#6-pull-requests)).
5. **Fill every section of the PR template**, including the cross-repo table.
6. **Expect review.** Run `/code-review` on large/numerical PRs; anything
   touching design, the DSL, or estimation also gets a human core-team review.
7. **Finish the [Definition-of-done checklist](sdlc.md#12-definition-of-done)**
   before marking it ready.

**Good first contributions:** add a NONMEM-comparison test for a model we don't
cover yet, improve a docs page, or fix a reproducible bug with a regression
test. Small, well-tested PRs are the easiest to merge. Large or experimental
features should be discussed in an Issue first and developed on a long-lived
feature branch rather than merged in half-working.
