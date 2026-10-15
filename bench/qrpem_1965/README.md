# Phase 0 benchmark: quasi-random importance sampling (epic #1965)

**Question.** Does quasi-random (Sobol) sampling in IMPMAP reduce Monte-Carlo
variability compared with pseudo-random IMPMAP and with SAEM? This is the
question a Phoenix-style QRPEM would answer.

**Setup.**

- **Run:** commit `28da032d`, release `ferx` CLI, Apple M5 Pro, 2026-10-14.
- **Parallelism:** 4 concurrent fits × 2 threads for the precision runs. A
  separate serial pass (1 fit at a time, 2 threads, 3 seeds) produced the wall
  times.
- **Common evaluator:** every fit's final estimates are re-scored by one
  evaluator (`method = imp`, `imp_eval_only = true`, K = 20 000,
  `imp_seed = 777`, t₅ proposal). The same seed for every arm gives common
  random numbers, so differences between arms are differences in the
  *estimates*. Its MC SE is 0.02–0.23, as listed in `summary.md`.
- **IMPMAP settings:** 200 iterations, 50 averaged, `impmap_auto = false`
  (fixed K).
- **Seeds:** 10 per arm (FOCEI is deterministic and runs once).

| model | η | N | data |
|---|---|---|---|
| `warfarin` | 3 | 10 | rich |
| `two_cpt_oral_cov` | 5 | 30 | rich, covariates |
| `three_cpt_iv` | 6 | 15 | rich |
| `sparse_combined` (`models/`, data `saem_sparse_combined.csv`) | 3 (block) | 300 | sparse, combined error |

Arms:

- `impmap_prng`: normal proposal, pseudo-random draws.
- `impmap_sobol`: normal proposal, Sobol draws.
- `impmap_t4`: the current default t₄ proposal, which silently cannot use Sobol.
- `saem`.
- `focei`.
- Diagnostics run on two_cpt only: `impmap_prng_mceta3`,
  `impmap_prng_defensive` (α = 0.1) and `impmap_prng_iter600`.

Files:

- `summary.md`: full precision table.
- `timing.md`: serial wall times.
- Raw runs are not committed. To regenerate:
  `python3 bench.py run --seeds 10 --eval`, then `python3 bench.py report`.

## Result 1: Sobol works, but only buys samples at small K

SD across seeds of the common-evaluator −2LL, and median relative SD of the
estimates (θ, Ω, σ):

| model | K | prng −2LL SD | sobol −2LL SD | prng param SD | sobol param SD |
|---|---|---|---|---|---|
| warfarin | 100 | 1.0e-4 | **2.3e-6** | 1.6e-4 | **3.5e-5** |
| warfarin | 300 | 1.8e-5 | **5.2e-7** | 1.8e-4 | **3.3e-5** |
| warfarin | 1000 | 3.8e-6 | **0** | 7.7e-5 | **3.3e-7** |
| sparse_combined | 100 | 0.096 | **0.020** | 0.044 | **0.018** |
| sparse_combined | 300 | 0.012 | 0.015 | 0.017 | **0.006** |
| sparse_combined | 1000 | 0.010 | 0.008 | 0.012 | 0.013 |
| three_cpt_iv | 100 | 0.016 | **0.007** | 0.0042 | 0.0024 |
| three_cpt_iv | 300 | 0.003 | 0.005 | 0.0021 | 0.0020 |
| two_cpt_oral_cov | any | 1.5–1.8 | 1.7–1.8 | 0.006 | 0.006 |

How to read the table:

- **At K = 100:** Sobol cuts the −2LL spread 2.4× (three_cpt_iv), 4.9×
  (sparse_combined) and 44× (warfarin) against pseudo-random draws at the same
  K. On the *parameters* it matches pseudo-random K = 300 on sparse_combined
  (0.018 vs 0.017), but not on the −2LL (0.020 vs 0.012); on warfarin it beats
  K = 1000 on both.
- **Cost:** wall time is roughly linear in K and Sobol costs the same per
  sample (`timing.md`). That is roughly a 3× cut in samples for the same
  precision at low K.
- **At K ≥ 300:** the two arms are tied, except on warfarin, where all of the
  numbers are negligible.
- **Dimension:** the gain shrinks with η dimension, as expected for
  quasi-Monte-Carlo (QMC).
- **Decision rule** (≥ 30% reduction in across-seed SD on the low-dimension
  models at equal K): **passes**. It passes on warfarin at every K, and on
  sparse_combined at K = 100 for the −2LL and at K = 100 and 300 for the
  parameters.

## Result 2: the largest variance source is MAP-mode trapping, which Sobol cannot fix

On `two_cpt_oral_cov`, every IMPMAP arm has an across-seed −2LL SD of about 1.7,
whatever K or point set it uses:

- **Two end points.** 4 of 10 seeds end at **the same** non-optimal point: −2LL
  −1196.15, TVV1 = 46.1, ω_KA = 0.22. The other 6 reach −1199.6.
- **Plateau, then a jump.** The seed-2 trace sits on a plateau around −1178
  (per-iteration IS objective) for 160 iterations, then jumps by 21 units.
- **Per-subject cause.** In the stuck runs, subject 6's warm-started MAP sits in
  a local mode whose EBE objective is 17.7 units worse than the global mode. A
  fresh inner solve at the same parameters finds the global mode.
- **Why IMPMAP never escapes.** The proposal is unimodal and centred on the
  wrong mode, so it never samples the right one, and θ is fitted around the
  wrong mode. This is the same ID-6 bimodality seen in #864 / #1504.

| arm (K = 300) | −2LL mean | −2LL SD | serial wall s |
|---|---|---|---|
| impmap_prng | −1197.80 | 1.74 | 4.22 |
| impmap_prng_defensive (α = 0.1) | −1197.24 | 1.65 | — |
| impmap_prng_iter600 | −1198.93 | 1.47 | — |
| **impmap_prng_mceta3** | **−1199.63** | **0.0036** | 4.79 |
| saem | −1199.10 | 0.84 | 0.12 |
| focei | −1199.61 | — | 0.06 |

How to read the table:

- **`impmap_mceta = 3`** (3 extra random MAP starts per subject per iteration)
  removes the spread: 480× smaller. It also gives the best −2LL of every arm,
  FOCEI included.
- **Cost of `mceta`** (serial, K = 300): +4.6% on sparse_combined, +13% on
  two_cpt, +28% on three_cpt_iv, +4% on warfarin.
- **What does not help.** The defensive mixture (which samples from the prior)
  and more iterations do not help.

## Phase 1b: Sobol on the default t₄ proposal

The Phase 1b binary makes `impmap_sobol` drive the Student-t scale (and the
defensive mixture) as well. Arm `impmap_t4_sobol` vs `impmap_t4`, 10 seeds, same
common evaluator:

| model | K | t₄ −2LL SD | t₄ + Sobol −2LL SD | t₄ param SD | t₄ + Sobol param SD |
|---|---|---|---|---|---|
| warfarin | 100 | 2.9e-4 | **1.9e-5** | 1.9e-4 | **8.3e-5** |
| warfarin | 300 | 4.2e-5 | **6.7e-7** | 1.0e-4 | **4.5e-5** |
| three_cpt_iv | 100 | 0.011 | **0.0023** | 0.0036 | 0.0024 |
| three_cpt_iv | 300 | 0.0034 | **0.0007** | 0.0020 | **0.0011** |
| sparse_combined | 100 | 0.014 | 0.011 | 0.018 | **0.0047** |
| sparse_combined | 300 | 0.0099 | **0.0050** | 0.011 | **0.0063** |

Unlike the normal proposal, the gain on t₄ holds at K = 300 and on the 6-η
model: 1.2–60× on the −2LL and 1.5–4× on the parameters. The per-iteration jitter
also drops (`summary.md`, tail jitter column).

The new binary reproduces every seeded non-Sobol and MVN-Sobol fit of the
Phase 0 binary to the printed digit (`bitid.py runs runs_bitid`; 9 fits).

## Other observations

- **Normal vs t₄ proposal.** The per-iteration IS objective jitters 3–10× more
  under the normal proposal than under the default t₄, with or without Sobol.
  The final estimates do not show the same gap, so this matters for
  convergence diagnostics, not for accuracy.
- **IMPMAP vs FOCEI and SAEM on sparse data.** On `sparse_combined`, IMPMAP
  reaches a 2.4 lower −2LL than FOCEI and 3.1 lower than SAEM, and its
  across-seed SD is 60–90× smaller than SAEM's (0.01 vs 0.89).
- **IMPMAP vs SAEM on cost.** IMPMAP is 30–100× slower than SAEM: 52 s against
  0.6 s on sparse_combined at K = 300.
  - Most of that cost is the weighted θ/σ M-step (profiled in Phase 3 below).
  - QRPEM uses only ~10 SIR-resampled draws per subject for this step
    (Phoenix "Number of SIR samples", default 10). Tested in Phase 2 and
    rejected: it is 2–6× faster but collapses a weakly identified σ.
- **Sample count.** K is not a power of two, but Sobol points balance best at
  K = 2^m. Tested in Phase 2: no gain on the IS fits (see below).

## Phase 2: Owen scrambling, and three ideas that lost

Same four models, t₄ proposal, K = 300, 200 iterations with the last 50
averaged, 10 seeds, scored by the common K = 20000 evaluator. Full table:
`summary_phase2.md`. The toggles for the rejected arms existed only in an
experiment binary; the harness keeps them as `PHASE2_ENV` for the record.

Spread of the final −2LL across seeds:

| model | pseudo-random | Sobol, shift | Sobol, Owen |
|---|--:|--:|--:|
| sparse_combined | 0.0099 | 0.0050 | 0.0037 |
| three_cpt_iv | 0.0034 | 0.00072 | 0.00040 |
| two_cpt_oral_cov | 0.0014 | 0.0021 | 0.00095 |
| warfarin (noise floor) | 4.2e-5 | 6.7e-7 | 2.0e-6 |

- **Owen scrambling: adopted.** It replaces the Cranley–Patterson shift
  outright, with no option to keep the shift: it beats the shift on every model
  above the warfarin noise floor, and on `two_cpt_oral_cov` the shift was worse
  than pseudo-random. On a smooth test integrand (the unit test
  `owen_scrambled_sobol_integrates_faster_than_shifted`) the RMSE at K = 1024 is
  3.1e-4 against 5.2e-3 for the shift.
- **Common random numbers (the same points every iteration): rejected.** The
  per-iteration objective goes smooth, but the fixed points bias the
  fit: the spread across seeds is 7–200× worse and the means move (e.g.
  `two_cpt_oral_cov` Sobol+CRN −1199.628 against −1199.652).
- **SIR M-step (10 resampled draws per subject, as in QRPEM): rejected.** It is
  2.3–6× faster (sparse_combined 76 s → 12 s), but on sparse_combined the
  additive σ collapsed to 0.0003 (from 1.63) on all 10 seeds and the mean −2LL
  is 0.31 worse, about 3× the evaluator's Monte Carlo error. 30 resamples still
  collapsed 1 seed in 10. Using SIR only during burn-in (full K while averaging)
  was fine on the other three models (1.5–2× faster) but still collapsed σ on 5
  of 10 sparse_combined seeds, and it never recovered. A weakly identified σ
  under a noisy M-step walks down the log scale.
- **Round K down to a power of two under Sobol (300 → 256): rejected.** On the
  smooth test integrand Owen at K = 256 beats K = 300 by 3.4×, but on the IS
  fits the spread was the same or slightly worse.

## Phase 3: the MAP step does not dominate

Profiled with macOS `sample` on `main` after #1997 (which re-solves every MAP
from a cold start too). Share of the fit thread's wall time:

| model | weighted M-step | IS draws | ISCALE | MAP |
|---|--:|--:|--:|--:|
| two_cpt_oral_cov | 70% | 19% | 10% | 8% |
| sparse_combined | 83% | 14% | 8% | 2% |
| three_cpt_iv | 70% | 17% | 9% | 11% |

The M-step is `theta_sigma_weighted_mstep`, BOBYQA re-scoring all K draws per
subject. Speeding up the MAP step would save at most 2–11%, so the Phase 3
gate fails and the MAP-skipping work was not started. The M-step is the lever,
and the cheap way to shrink it (SIR, above) is not safe.

## Phase 4: should `impmap_sobol` default to on?

Default IMPMAP settings (`impmap_auto` grows K from 300 up to 10000), with and
without `impmap_sobol = true`, 10 seeds. Full table: `summary_phase4.md`.

| model | spread, pseudo-random | spread, Sobol | evaluator MC error |
|---|--:|--:|--:|
| sparse_combined | 0.0044 | 0.0025 | 0.10 |
| three_cpt_iv | 0.00012 | 0.00005 | 0.049 |
| two_cpt_oral_cov | 0.00050 | 0.00078 | 0.24 |
| warfarin | 4.2e-5 | 2.0e-6 | 0.016 |

At these settings the wall time is dominated by how long `impmap_auto` runs,
and the timings above could not separate the arms: serial and alternating, but
on a shared machine (load 5–11), `three_cpt_iv` took 71/54/82 s pseudo-random
against 49/46/54 s Sobol over seeds 1–3, and `sparse_combined` 202/484 s against
258/488 s over seeds 1–2. At fixed K the cost is measurable (next section).

**Decision: keep the default off.** At default settings the spread across seeds
is already 20–500× below the Monte Carlo error of the −2LL itself, so halving it
buys nothing a user can see, and on `two_cpt_oral_cov` Sobol widened it. Turning
it on by default would change every seeded IMPMAP fit and re-pin every anchor
for that. Sobol stays the recommendation for a fixed, small K.

## Old vs new Sobol on an idle machine

The merge-base binary (`ac14b56a`, Cranley–Patterson shift) against this
branch (Owen), at t₄, K = 300, 200 iterations, 10 seeds. One fit at a time on 2
threads, binaries alternated seed by seed, load 4–5 (an idle desktop plus the
fit). Tables: `summary_ovn_old.md`, `summary_ovn_new.md`.

| model | spread, shift | spread, Owen | wall s, shift | wall s, Owen |
|---|--:|--:|--:|--:|
| sparse_combined | 0.0050 | 0.0037 | 60.6 | 60.8 |
| three_cpt_iv | 0.00072 | 0.00040 | 4.13 | 4.09 |
| two_cpt_oral_cov | 0.0021 | 0.00095 | 4.76 | 4.78 |
| warfarin | 6.7e-7 | 2.0e-6 | 1.31 | 1.31 |

- The spreads reproduce the Phase 2 numbers to every printed digit: the fits
  are deterministic given the seed.
- Owen costs nothing over the shift (medians within 0.5%).
- Control: the pseudo-random arm (seeds 1–3) gives byte-identical fit YAML on
  both binaries apart from the wall-time fields, so Sobol is the only change.
- **Sobol itself is not free at fixed K.** In both binaries the Sobol fits ran
  8–18% longer than the pseudo-random ones over the same seeds 1–3:
  sparse_combined 51 → 60 s, three_cpt_iv 3.8 → 4.1 s, two_cpt_oral_cov
  4.3 → 4.8 s, warfarin 1.13 → 1.33 s. This predates this branch; it was not
  profiled.
