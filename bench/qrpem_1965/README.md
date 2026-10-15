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
  - A likely large share of that cost, **not yet profiled**, is the weighted
    θ/σ M-step: `theta_sigma_weighted_mstep` re-scores all K draws for every
    subject on every BOBYQA evaluation.
  - QRPEM instead uses only ~10 SIR-resampled draws per subject for this step
    (Phoenix "Number of SIR samples", default 10).
  - This is untested here, but it is the most promising speed lever.
- **Sample count.** K is not a power of two, but Sobol points balance best at
  K = 2^m. This is a free improvement for Phase 1 or Phase 2.
