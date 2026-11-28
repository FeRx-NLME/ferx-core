# Peak-memory probe

The compiled-plan regression of \#2939 took one evaluation of a 9-state stiff PK/PD model from 86 MB to 3–4 GB, and a full fit to 11 GB. No job tracked peak memory, so it was found by hand, after the fact. The peak-memory probe closes that gap (#3081).

|  |  |
|----|----|
| Model | `tools/peak-memory/stiff_pkpd.ferx`, 100 subjects in `stiff_pkpd.csv` |
| Runner | `tools/peak-memory/peakmem.py` (`run`, `report`, `accept`), standard-library Python only |
| Baseline | `tools/peak-memory/baseline.json` |
| Workflow | `.github/workflows/peak-memory.yml`, nightly 05:00 UTC and on dispatch, **Linux only** |
| Output | the job summary table, and a `peak-memory-results` artifact kept 90 days |
| Guard | `tests/peak_memory_probe.rs` (per PR, runs no fit) |

## The model

The model that regressed (e7070) is private, so the probe is a synthetic model of the same shape. The shape is what drives plan memory, not the drug:

- **9 nonlinear ODE states**: a two-compartment oral PK with saturable (Michaelis–Menten) elimination, an effect compartment, and a Friberg transit-chain myelosuppression PD with the `(BASE/circ)^GAM` feedback.
- **Stiff**: the fast effect-compartment and distribution rates against a five-day maturation chain make `ode_method = auto` escalate every dosing segment to a stiff stepper.
- **10 random effects**, two endpoints (PK and PD), and two cycles of five daily doses, so every subject carries ten event segments.

The data were simulated from the model at its initial estimates:

``` bash
cargo run --release --example generate_peak_memory
```

## What a nightly does

1.  Builds the **shipped** binary, `cargo build --release -p ferx-cli`: the release profile with fat LTO, and mimalloc as the global allocator. The other nightlies use the `ci-test` profile, which is not what users run.
2.  Runs two cases under GNU `/usr/bin/time -v`, each on 4 threads:
    - `eval`: one evaluation at the initial estimates (`maxiter = 0`);
    - `fit`: one outer iteration (`maxiter = 1`, an NLopt budget of n + 1 evaluations), then the covariance step. Each case is the model file with its `maxiter` and `covariance` lines rewritten. The model file itself carries the `eval` spelling.
3.  Compares each case’s peak resident set size with the baseline and writes the table to the job summary.

The thread count is fixed because the per-thread pools are part of what is measured. A baseline is only comparable at the thread count it was taken at. `peakmem.py` rejects a fit whose `n_threads_used` does not match.

## Verdicts

| Glyph | Meaning | Job |
|:--:|----|----|
| `=` | within ±25 % of the baseline | passes |
| `▲` | more than 25 % above the baseline | **fails** |
| `▼` | more than 25 % below the baseline: lock it in with `accept` | passes |
| `!` | the probe drifted: `auto` stopped escalating, the outer gradient is not recorded as analytic (a missing field counts as drift), a compiled plan fell back to finite differences (`W_PLAN_FD_FALLBACK`, including in the covariance step, which still reports `Computed`), or a case’s covariance status is not the expected one (`NotRequested` for `eval`, `Computed` for `fit`) | **fails** |
| `✗` | the case failed, timed out, or was not run | **fails** |
| `·` | not judged: another platform, thread count or input files than the baseline | passes |

The tolerance is `TOLERANCE` in `peakmem.py`. The regression this exists for was 10–40×, so the band only has to clear run-to-run noise.

A drifted probe fails even when its peak is lower. If a change routes the model off the compiled plans, stops the stiff escalation, or skips or breaks the `fit` case’s covariance step (where e7070’s post-fit spike lived), the peak can drop for a reason that has nothing to do with memory. That is a different measurement, not a win, and the probe has to be repaired before its numbers mean anything.

A baseline records the **runner** it was measured on: `github-hosted` for the nightly, `local` for anything outside GitHub Actions. A baseline from anywhere but `github-hosted` is **provisional**. A laptop, a container or emulation can report the same `Linux/x86_64` platform string while moving peak RSS by more than the tolerance, so the first `github-hosted` run against a provisional baseline fails the job, whatever its verdicts, until its artifact is accepted. A local run against it is judged as usual.

A baseline taken on another platform is shown but not judged. Linux is the reference platform (see [Reference platform](../development/sdlc.llms.md#reference-platform-linux)), and the allocator, libc and architecture move peak RSS by more than the tolerance. The same holds when `stiff_pkpd.ferx` or `stiff_pkpd.csv` no longer hash to the values the baseline records. The baseline then measured a different thing.

### Sensitivity

The probe was fed the failure it exists to catch. Peak RSS on macOS arm64, 4 threads, `ci-test` builds:

| case   | `4350b315` (before \#3076 / \#3078) | `64b3afd7` | ratio |   wall |
|--------|------------------------------------:|-----------:|------:|-------:|
| `eval` |                             1015 MB |     114 MB |  8.9× |   72 s |
| `fit`  |                             3831 MB |     237 MB |   16× | 25 min |

`4350b315` is main just before \#3076 and \#3078 trimmed plan memory. The OFV is the same on both builds. These are macOS numbers, so they show only the ratio. The baseline is a Linux measurement.

## The baseline and its ratchet

`baseline.json` records, for each case, the peak RSS in bytes, plus the platform, the thread count, the SHA-256 of the model and the data, the run it was measured on, and a reason. It follows the same ratchet idea as `docs/.lint-baseline`. The nightly never edits it. It moves only in a commit that says why:

- **Down**, when the report shows `▼`: an improvement that is not recorded can be spent by the next regression without failing the job.
- **Up**, when a change genuinely needs more memory. The PR says what the memory buys.
- **Re-measured**, whenever the model or the data change, in the same PR.

To move it, dispatch the workflow (on `main`, or on the branch whose change moves the peak), download its results, and accept them:

``` bash
gh workflow run peak-memory.yml --ref <branch>
gh run download <run-id> --name peak-memory-results --dir pm
python3 tools/peak-memory/peakmem.py accept --results pm/results.json \
  --reason "#1234: plan logs checkpointed; eval 610 → 140 MB"
```

`accept` refuses a run with a failed case, a drifted probe (including a `fit` whose covariance step did not report `Computed`), and an empty reason. `tests/peak_memory_probe.rs` fails when the committed baseline’s hashes do not match the committed model and data, or when it was not taken on Linux.

### The committed baseline

The baseline is the nightly of 2026-11-24 (run 45331581482, `main@c75687b4`), runner `github-hosted`. The four nightlies before it agree to within 4 %:

|  | eval RSS | fit RSS | eval wall | fit wall |
|----|---:|---:|---:|---:|
| provisional (OrbStack Rosetta, `local`) | 205 MB | 962 MB | 94 s | 1420 s |
| nightlies 2026-11-21 to 24 | 160–167 MB | 375–391 MB | 294–629 s | 3147–5049 s |
| **baseline** (45331581482) | **162 MB** | **391 MB** | 621 s | 4807 s |
| native Linux arm64, release | 154 MB | 361 MB | 72 s | 924 s |

The first baseline was taken under emulation, before the workflow could be dispatched. Emulation inflated the `fit` peak 2.5×, so the nightlies failed as provisional until one was accepted (#3081). Every row reaches the same OFVs, to the last digit.

## Wall time and timeouts

The probe measures memory, but it has to finish to measure anything. A github-hosted runner is slow and noisy for this workload. It is 5–9× slower than a native arm64 run and moves 2× from one night to the next, while doing the same computation, down to the same OFV bits. The cause is the runner. It is not subnormal arithmetic: the evaluation sets no underflow flag on any worker thread.

On the native run, the `fit` case’s time splits into estimation, 628 s, and covariance, 288 s. Estimation is 82 s of cold inner solves, 24 warm NLopt evaluations at about 20 s each, and 64 s for the final inner loop. In the `eval` case, the inner EBE searches take about 80 % of the time: 209 analytic BFGS gradient calls per subject from a cold start, at about 10 ms each.

So each case has its own timeout, `CASE_TIMEOUT_S` in `peakmem.py`:

| case   | timeout | slowest night | margin |
|--------|--------:|--------------:|-------:|
| `eval` |  1800 s |         629 s |   2.9× |
| `fit`  |  8100 s |        5049 s |   1.6× |

The job’s `timeout-minutes` covers both plus the build, so a case that times out is still recorded and uploaded rather than cut off by GitHub. The guard test pins that inequality. The report adds a timing table: wall time, estimation and covariance as the fit measured them, and the share of the timeout. A case past 80 % of its timeout gets a `⚠`. That is a warning, not a failure.

For a slow night, the artifact holds what is needed without a rerun:

- `<case>.stderr.log`: the case’s stderr, each line stamped with seconds since the start, kept on a timeout too;
- `<case>-fit.json`: the fit JSON, with every warning;
- `results.json`: the host’s `cpu_model` and `cpu_count`, since hosted runners land on more than one CPU model.

If the margin erodes, the cheaper fix is fewer subjects. That changes the data, so it means re-measuring the baseline and showing again that the probe catches \#2939.

## Running it locally

``` bash
cargo build --release -p ferx-cli
python3 tools/peak-memory/peakmem.py run --ferx target/release/ferx --out results.json
python3 tools/peak-memory/peakmem.py report --results results.json
```

On macOS the script reads BSD `time -l` instead of GNU `time -v`, and the report shows the numbers without judging them. For a breakdown of *where* the memory goes, profile with a build that uses the system allocator: mimalloc attributes whole segments to whichever call grew them.
