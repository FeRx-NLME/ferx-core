# VI cross-implementation comparison — ferx vs nlmixr2 `emvi`

The harness for **Anchor B** of `VI_VALIDATION.md`. nlmixr2 7.0's `est = "emvi"` is the only
other variational-inference implementation in a pharmacometrics package, which makes it the only
external check on ferx's VI *as VI* — every other comparator (NONMEM, FOCEI, SAEM) targets the
same parameters by a different approximation, so a disagreement is always arguably bias rather
than a bug.

Read `VI_VALIDATION.md` §4 before using this. In particular §4.3 (four `emvi` defaults that
would confound the comparison) and §4.11 (what the first run found).

## Running it

```bash
tools/vi-emvi-comparison/run.sh      # from the repo root
```

First run installs nlmixr2est + rxode2 + RcppParallel as prebuilt binaries into an **isolated**
R library, leaving your system R library untouched. Everything persistent lives in
`~/.local/share/ferx-vi-validation` (override with `FERX_VI_STATE`) — the R library is ~110 MB,
which is why it is not in the repo. Deleting that directory undoes the install completely.

Not wired into CI, and shouldn't be: no CI image here carries an R stack, and the fits take
minutes.

## Contents

| File | Role |
|---|---|
| `run.sh` | driver — installs, sets the linker workaround, runs both sides |
| `emvi-compare.R` | nlmixr2 side: FOCEI (convention anchor), `emvi`, `emvi` with `returnVi` |
| `warfarin_cmp.ferx` | ferx FOCEI, same model |
| `vi_adam.ferx` | ferx VI, `vi_omega_update = adam` |
| `vi_closed_form.ferx` | ferx VI, `vi_omega_update = closed_form` |

All five describe the *same* model — 1-cpt oral, lognormal `η` on CL/V/KA, proportional error,
matching the `tests/vi.rs` fixture. nlmixr2's mu-referenced form is equivalent:
`exp(log(TVCL) + η) == TVCL·exp(η)`, which is why its initial estimates are logs of ferx's.
Both sides use a closed-form solution (`linCmt()` / `one_cpt_oral`) so an ODE-vs-analytic
difference cannot contaminate the result.

## Why FOCEI runs first

ferx's objective omits `½·n_obs·log 2π` (the NONMEM convention) and nlmixr2 has an `adjObf` flag
that defaults to `TRUE`. Rather than reason about whether they agree, **measure it**: run FOCEI on
both and compare. The offset is whatever it is, and it applies to the VI comparison too.

Measured on warfarin: `−286.004` (ferx) vs `−285.947` (nlmixr2), i.e. **zero offset** — a
convention mismatch would have shown as `110 × log 2π ≈ 202`. The by-product is most of Anchor D,
the standing NONMEM-comparison placeholder in `docs/estimation/vi.qmd`.

## Two things to know before trusting a number

**Both `Ω` routes, deliberately.** `emvi` steps `Ω` by gradient; ferx's default takes the exact
ELBO maximizer in closed form. Those coincide at the optimum (the closed form is where the
gradient vanishes), so both ferx arms are run and agreement of *both* with `emvi` is the real
result. First run: the two arms agree with each other to under 0.5%.

**`emvi` is explicitly a research method.** Its own release notes say it is "not validated to the
standard of the established estimation methods" and may be "withdrawn in a future release without
a deprecation cycle". So this is **co-validation, not anchoring**: independent code, independent
authors, same mathematics — agreement raises confidence a lot, but neither side is ground truth.
The only item here with actual ground truth is Anchor A (`src/estimation/vi/elbo_agq_bound.rs`).

## Known limitation

The per-subject Cholesky packing in nlmixr2's `Lpack` is **not yet proven** row- vs column-major
(`VI_VALIDATION.md` §4.7). Row-major reproduces plausible variances and column-major does not, so
row-major is very likely right — but until it is settled, only `S[1,1]` should be compared
against ferx, since it equals `L[1,1]²` under either convention. `emvi-compare.R` says so at the
point of use.

## The macOS linker trap

If R's `FLIBS` points at `/opt/gfortran` and the CRAN gfortran was never installed, **every**
rxode2 model fails — at the *link* step, while the C compile succeeds, which makes it look like a
code or version problem. It is neither, and it is not specific to the isolated library: an older
system rxode2 fails identically.

`run.sh` handles it by pointing `R_MAKEVARS_USER` at a one-line `Makevars` with an empty `FLIBS`,
scoped to the process. That works because macOS links with `-undefined dynamic_lookup`, so the
Fortran symbols reached through `-lRblas`/`-lRlapack` resolve from R's own bundled libraries at
load time. Installing the CRAN gfortran is the alternative, and needs `sudo`.
