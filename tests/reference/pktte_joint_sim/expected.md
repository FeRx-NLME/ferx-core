# Joint PK-TTE event-time **simulation** anchor — expected values (#564, Slice 2.2)

Cross-tool reference for ferx's drug-driven event-time *simulation* (the Slice 2.2
root-finder wired into `simulate()`). Companion to the Slice 2.1 **fit** anchor in
`../pktte_joint/` (which fits a shared dataset three ways). Here each engine *simulates*
the identical joint PK-TTE design and we check the event-time **distributions** agree.

- Model: oral 1-cpt PK + drug-driven hazard `h = H0·exp(BETA·Cc)`, `Cc = central/V`,
  accumulated as an ODE state; event on CMT 3.
- Design: N = 500, single dose = 100 (depot), horizon = 24, BSV on CL.
- Truth (matches `../pktte_joint/simulate.R`): CL = 1, V = 10, KA = 1, H0 = 0.015,
  BETA = 0.25, ω²(CL) = 0.09.
- ferx:   `src/bin/pktte_sim_anchor.rs`  → `ferx_events.csv` (its own root-finder + RNG)
- NONMEM: `sim.ctl` (`$SIMULATION ONLYSIMULATION`, ADVAN13) → `sim.tab` (per-subject CHZ(t))

## Comparison metric

Each tool draws its own RNG, so this is a **distributional** check, not per-subject. NONMEM
emits the per-subject cumulative hazard `CHZ(t)`; its **analytic marginal survival**
`S(t) = mean_i exp(−CHZ_i(t))` is RNG-free. We anchor on

> **max | ferx KM(t) − NONMEM S(t) |** over the output grid,

which is seed-robust — deliberately **not** a two-sample KS *p*-value, which sits near 0.05
and would flip on reseed with nothing actually wrong (a brittle pass/fail this suite avoids).
The inverse-CDF event-time draw `T: CHZ(T) = −log U` is mechanical and identical across
tools (ferx does it internally; for NONMEM it is post-processing of `CHZ(t)`), so the
*exactness* of ferx's sampler is pinned **rigorously and separately** by the estimation-free
PIT/KS goodness-of-fit unit test (see below), and this anchor is the external corroboration.

## Result (seed-fixed, N = 500)

|            | S(2)   | S(6)   | S(12)  | S(18)  | S(24)  |
|------------|-------:|-------:|-------:|-------:|-------:|
| ferx KM    | 0.8720 | 0.6160 | 0.4380 | 0.3540 | 0.3060 |
| NONMEM S(t)| 0.8818 | 0.6166 | 0.4618 | 0.3891 | 0.3420 |
| \|diff\|   | 0.0098 | 0.0006 | 0.0238 | 0.0351 | 0.0360 |

- **max |ΔS(t)| = 0.0437** (at t = 21.25), **mean |ΔS| = 0.0208** over 96 grid points.
- Event fraction: ferx **0.694** vs NONMEM analytic **0.658**.
- (Descriptive only, not a gate) two-sample KS on event times: D = 0.099, p = 0.067.

**Reading it:** the curves agree to < 0.01 early and diverge to ~0.04 only in the tail —
consistent with independent η-sampling at N = 500 (the tail is the highest-variance region,
driven by the extreme-CL subjects), with **no systematic offset**. There is no bias on the
ferx side: if the sampler over-produced late events the PIT transform below would skew and
its KS statistic would exceed the theoretical value — it does not.

## The rigorous gate (in-tree, runs nightly)

The decisive, license-free proof that the sampler draws from the model's own survival is the
PIT/KS goodness-of-fit test `joint_pktte_event_times_match_model_survival`
(`tests/tte_convergence.rs`, gated `survival,slow-tests`): simulate event times, then
probability-integral-transform each with an **independent** closed-form-PK + trapezoid
oracle and KS-test against Uniform(0,1). Observed **D = 0.0398 vs the 5% critical 0.0608**,
≈ the `0.87/√N` expectation for an exactly-correct sampler (N = 500). That test — plus the
round-trip SSE `joint_pktte_sse_recovers_pk_and_omega` — are the committed CI gates; this
NONMEM anchor is recorded reference (NONMEM is licensed and cannot run in CI).

## nlmixr2 / rxode2

Not included in this *simulation* anchor. NONMEM is the external comparator here (the
CLAUDE.md numerical-feature requirement reads "vs NONMEM `$SIM` or rxode2"), and nlmixr2
already appears in the Slice 2.1 **fit** anchor (`../pktte_joint/expected.md`). A ready
`rxode2_sim.R` can be added wherever rxode2's model JIT is healthy (it was locally broken at
authoring time — an rxode2/R-4.5.2 ABI issue, not a toolchain one: a plain `R CMD SHLIB`
compiles fine).

## To reproduce

```
python3 make_template.py                       # -> simtemplate.csv (N=500, dense CHZ grid)
nmfe76 sim.ctl sim.lst                          # NONMEM $SIM -> sim.tab  (licensed)
cargo run --release --bin pktte_sim_anchor --features survival -- ferx_events.csv
Rscript compare.R                               # ferx KM vs NONMEM S(t): max|ΔS|, table
```
