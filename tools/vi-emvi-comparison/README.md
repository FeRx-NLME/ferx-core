# VI cross-implementation comparison — ferx vs nlmixr2 `emvi`

The harness for **Anchor B** of `VI_VALIDATION.md`. nlmixr2 7.0's `est = "emvi"` is the only
other variational-inference implementation in a pharmacometrics package, which makes it the only
external check on ferx's VI *as VI* — every other comparator (NONMEM, FOCEI, SAEM) targets the
same parameters by a different approximation, so a disagreement is always arguably bias rather
than a bug.

Read `VI_VALIDATION.md` §4 before using this. In particular §4.3 (four `emvi` defaults that
would confound the comparison), §4.11a (**what the first run got wrong**) and §4.14 (the
corrected results).

> **The first run of this harness measured its own settings, not the estimators.** `emvi` at
> `iters = 2000` was not converged, and ferx had a `σ` step-size defect. Both are fixed —
> `iters` now defaults to 20000, the `.ferx` arms run `vi_mc_samples = 128`, and ferx carries a
> closed-form `σ` M-step — and the conclusions changed materially: the OFV gap to FOCEI went
> from ~10 units to under 1.3, and `emvi`'s apparent 24× per-subject variance spread to 2.9×.
> The lesson is in §4.11a: a two-tool comparison localizes a disagreement but cannot orient one,
> so `agq_ref.ferx` is now the arbiter and the per-subject figures are read against it.

## Running it

```bash
tools/vi-emvi-comparison/run.sh              # from the repo root: both sides, all fits
Rscript tools/vi-emvi-comparison/plots.R     # then the four figures
```

`plots.R` keeps its input and output directories separate, because the figures are usually
wanted somewhere a person actually looks while the fits stay in the harness state dir:

| Variable | Role | Default |
|---|---|---|
| `FERX_VI_OUT` | where the fits are **read** from | `$FERX_VI_STATE/results` |
| `FERX_VI_FIGS` | where the figures are **written** | same as `FERX_VI_OUT` |

So to send the figures elsewhere, set only `FERX_VI_FIGS` — the directory is created if needed:

```bash
FERX_VI_FIGS=./figures Rscript tools/vi-emvi-comparison/plots.R
```

Pointing `FERX_VI_OUT` at the destination instead is the mistake to avoid: the script would look
for `emvi-results.rds` in the figure folder and stop with a named-input error.

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
| `emvi-compare.R` | nlmixr2 side, both models: FOCEI (convention anchor), `emvi`, `emvi` with `returnVi`, then the three mixed-Ω fits |
| `plots.R` | the four figures, drawn from both sides' saved output — run after `run.sh`, fits nothing; `FERX_VI_FIGS` sets where they land |
| `warfarin_cmp.ferx` | ferx FOCEI, diagonal Ω |
| `vi_adam.ferx` | ferx VI, `vi_omega_update = adam` |
| `vi_closed_form.ferx` | ferx VI, `vi_omega_update = closed_form` |
| `warfarin_block_cmp.ferx` | ferx FOCEI, **mixed** Ω — read its header before comparing its Ω |
| `vi_block_adam.ferx` | ferx VI, mixed Ω, `adam` route |
| `vi_block_closed_form.ferx` | ferx VI, mixed Ω, `closed_form` route |
| `vi_block_mean_field.ferx` | ferx VI, mixed Ω, `vi_family = mean_field` |

The first five describe the *same* model — 1-cpt oral, lognormal `η` on CL/V/KA, proportional error,
matching the `tests/vi.rs` fixture. nlmixr2's mu-referenced form is equivalent:
`exp(log(TVCL) + η) == TVCL·exp(η)`, which is why its initial estimates are logs of ferx's.
Both sides use a closed-form solution (`linCmt()` / `one_cpt_oral`) so an ODE-vs-analytic
difference cannot contaminate the result.

The last four swap that diagonal Ω for a **mixed** one — `block_omega (ETA_CL, ETA_V)` plus a
standalone `ETA_KA`, the same shape as `src/estimation/vi/run_tests.rs::mixed_omega_fixture` —
and exist for the two claims in `VI_VALIDATION.md` §4.9 that no other comparator reaches. Skip
them with `FERX_VI_DIAG_ONLY=1` when only the §4.11 diagonal numbers are wanted.

## Why FOCEI runs first

ferx's objective omits `½·n_obs·log 2π` (the NONMEM convention) and nlmixr2 has an `adjObf` flag
that defaults to `TRUE`. Rather than reason about whether they agree, **measure it**: run FOCEI on
both and compare. The offset is whatever it is, and it applies to the VI comparison too.

Measured on warfarin: `−286.004` (ferx) vs `−285.947` (nlmixr2), i.e. **zero offset** — a
convention mismatch would have shown as `110 × log 2π ≈ 202`. The by-product is most of Anchor D,
the standing NONMEM-comparison placeholder in `docs/estimation/vi.qmd`.

## The mixed-Ω arms (§4.9)

**Claim (a) — structural zeros.** `docs/estimation/vi.qmd` tells users that in a mixed Ω the
*undeclared* off-diagonals are held "at exactly zero rather than letting them pick up sampling
correlation", for both `vi_omega_update` routes. `fbvi` errors outright on a correlated block,
so `emvi` is the only external VI that estimates one at all — which makes it the only outside
check that exists. `run.sh` reads the ferx side off the fit YAML by *absence*: `io/output.rs`
emits an `ETA_x__ETA_y:` entry for every pair with `|cov| > 1e-15`, so no entry means the slot
held, and the block's own `ETA_V__ETA_CL` entry being present is what proves the emitter ran.

**Confirmed on both sides, and unaffected by the correction.** All three ferx VI arms held both
zeros — `adam`, `closed_form`, `mean_field` — while estimating the block covariance, and all
three nlmixr2 mixed-Ω fits returned both slots as exactly `0e+00` (FOCEI, `emvi` fullRank, `emvi`
meanField). A structural zero is held or it is not, so this is the one verdict a convergence
problem could not have moved. It is also the one claim in `VI_VALIDATION.md` that only `emvi`
could reach.

Converged, the block covariance agrees across tools to **3%** — `0.001900` (ferx VI) against
`0.001958` (`emvi`), where the under-converged first run had them 29% apart. Read it as weak
evidence either way: ferx's FOCEI puts an SE of `0.004939` on that covariance, 2.6× the estimate,
so on 10 subjects the parameter is indistinguishable from zero. Both sides imply a correlation of
`0.11`.

**A found asymmetry, worth knowing before reading any Ω across tools.** ferx's *FOCEI* does not
honour the declared structure: on `warfarin_block_cmp.ferx` it returned `cov(KA,CL) = −0.030329`
and `cov(KA,V) = 0.019807`, and reported `n_parameters = 10` — 3 θ + a full 6-element lower
triangle + 1 σ, where an honoured mixed Ω is 8. VI, SAEM and GN all filter on
`omega.free_mask`; the outer optimizer's pack/unpack (`estimation/parameterization.rs`)
enumerates the whole lower triangle, and only the covariance step consults
`omega_structural_zero_mask`. nlmixr2's FOCEI *does* honour the block. So the FOCEI arm is a
full-Ω reference, and a cross-tool Ω comparison across the `ETA_KA` row compares two different
models. (`n_parameters = 10` is also reported by the VI arms, which do hold the zeros, so AIC
and BIC there are penalising two parameters that were never estimated.)

**Claim (b) — `mean_field` vs `full_rank` bound looseness.** `vi.qmd` states that a diagonal `q`
costs "a looser bound whenever the true posterior is correlated"; that rests on theory alone,
and §4.9 makes `emvi` the external check because the families match by name
(`viFamily = "meanField"`).

**Measured on the ferx side, directional on nlmixr2's.** The first run of this arm could only
give a direction, because each family runs its own M-step and the two arms landed at different
`(θ, Ω, σ)` — an ELBO bounds `−2 log L` *at the parameter vector where it was evaluated*, so a
gap between two different vectors attributes nothing. Converged, that confound is gone on the
ferx side: the two arms now agree to a **0.89% max parameter drift** (θ ≤ 0.005%, Ω ≤ 0.37%, σ
0.89%), inside the 2% threshold `emvi-compare.R` enforces, and `mean_field`'s bound is **7.70
units looser** (`−2·ELBO` `−277.873` against `−285.576`). That is a measurement, and it is the
first evidence for a `vi.qmd` claim that had rested on theory alone.

`emvi`'s arms still drift 5.73% on the fixed effects, so its side stays directional — but it
agrees on the sign (`meanField` ELBO `122.968` against `fullRank`'s `126.171`). The harness
reports the drift beside the ELBO gap and withholds a verdict above 2%, so it says which of
these two situations you are in.

Two traps here. `elbo_tightness_ratio` is **not** the quantity to read — it is
`excess / Σ(dᵢ/2)`, a health diagnostic (`elbo.rs:1067`) whose denominator is `d = 3` for both
families. And the one thing that *does* work in the comparison's favour: `meanField`-vs-
`fullRank` is an ordering *within* one implementation, so §4.11's unreconciled ELBO scale
cancels out of the difference.

Closing claim (b) properly needs both families' ELBO at **one** parameter vector, which no CLI
option reaches — it is a Tier-1 unit test over `elbo_agq_bound.rs`, in the shape
`family_tests.rs` already uses when it loops the families over a fixed fixture.

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
