# `recorded-value-read` — a value read on a generic number outside a non-recording arm

Rule of `crates/patterns-lint` (#2786). Gate: `tools/preflight.sh patterns` (CI job
`Patterns lint`). Print this file with
`cargo run -p patterns-lint -- --explain recorded-value-read`.

## Why

The compiled per-subject plans record the generic `*_g<T: PkNum>` kernels over the recording
type `Tr` (`parser/kernel_jit.rs`). A recording must never *read* a value: the value of an
input-dependent number is not a constant of the plan, so a decision taken on it is either baked
in (wrong at the next point) or refused (the subject falls back to finite differences with
`W_PLAN_FD_FALLBACK`).

The first line of defence is the type system. `PkNum` has **no value accessor**: no `val()`,
no `seen_value()`. A kernel written over `T: PkNum` cannot read a value, and it decides through
the branch API (`br_cmp`, `br_cmp_c`, `br_abs_lt`, `br_absdiff_lt`, `br_is_nan`, `br_shift_cmp`,
`select_if`), which `Tr` records as guards a plan re-evaluates. Value access lives on
`NonRecording: PkNum` (`f64`, the forward duals, `Taylor`, and a nested dual over one), which
`Tr` does not implement.

What is left is `NonRecRead::plain` (`x.plain()`) and `RecHooks::probe_value`, two reads. `plain` is a convenience for code that runs only for
plain types: it reads the value of a plain number and, on an input-dependent recorded value,
*fails the recording* instead of returning it. `probe_value` hands a recording's value over **unguarded** and is only for an `f64` evaluation whose result is checked not to depend on it (a slot constant in θ and η, an exact log-sum-exp shift); every use carries a `value-read-ok` marker saying so. The rule keeps both out of any code a
recording can reach.

## What it flags

Production code of `src/` only. In a function generic over a number type (a type parameter
bounded by `PkNum`, `VmNum`, `RecHooks`, `impl PkNum` in argument position, or a signature or
`impl` target naming `Tr`), every zero-argument `.plain()` / `.probe_value()` call and every `X::plain` / `X::probe_value` path. A function *named* `plain` or `probe_value` is the primitive and is skipped.

The analysis is syntactic: it takes a `.plain()` in such a function to be on a number-type value
(no other type has the method).

## Not flagged

- **A non-recording arm**: `if !T::RECORDS_BRANCHES { .. }`; the `else` of
  `if T::RECORDS_BRANCHES { .. }`; the right operand of `!T::RECORDS_BRANCHES && ..` and of
  `T::RECORDS_BRANCHES || ..`; the statements after `if T::RECORDS_BRANCHES { return .. }` (the
  body must end in `return`/`break`/`continue`). `!`, `&&` and `||` are evaluated, so
  `if T::RECORDS_BRANCHES && x { .. } else { .. }` exempts neither arm.
- A `#[cfg(test)]` / `#[test]` item, or a test file.
- A function whose doc comment has a line starting `Non-recording: <reason>`.

## Escape hatch

`// value-read-ok: <reason>` on a line of the call, or alone on the line above. The reason is
required. Use it only for a read that cannot affect recorded control flow (a finiteness check
that only decides whether to decline, a scale no result depends on).

## The baseline

This rule starts at zero: nothing is recorded for it in `baseline.txt`, and nothing may be
added. A finding is fixed (branch API, a `_g` form, or restructuring so the read sits in a
non-recording arm) or marked with a concrete reason.
