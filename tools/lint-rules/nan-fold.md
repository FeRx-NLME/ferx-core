# `nan-fold` — a `max`/`min` fold in test code swallows `NaN`

Rule of `crates/patterns-lint` (#2634). Gate: `tools/preflight.sh patterns`
(CI job `Patterns lint`). Print this file with
`cargo run -p patterns-lint -- --explain nan-fold`.

## Why

`f64::max` and `f64::min` return the **other** operand when one is `NaN`:
`0.0f64.max(NaN)` is `0.0`, `f64::INFINITY.min(NaN)` is `inf`. So

```rust
let mut worst = 0.0_f64;
for (got, want) in rows {
    worst = worst.max((got - want).abs());
}
assert!(worst < 1e-6);
```

passes on the strength of the rows that worked. A solver regression that returns
`NaN`, which is the likeliest way to break the thing an anchor anchors, goes
green. `pktte_tdep_nonmem_anchor` had five accumulators like this when it was
written (AGENTS.md, "A green test is not evidence that it can fail"). Before this
rule the hazard was documented in prose only.

## What it flags

Test code only: `tests/**`, `crates/*/tests/**`, `src/**/*_tests.rs`,
`src/**/tests/**`, any file reached through a `#[cfg(test)] mod x;` declaration,
and `#[test]` / `#[cfg(test)]` items in any other file. A `cfg(any(test, …))`
item also compiles into production, so it does not count. Production folds are
out of scope: most of the ~450 there are correct, such as a running maximum of
step sizes or a tolerance floor.

| Shape | Example |
|---|---|
| accumulator | `worst = worst.max(err)`, `lo = lo.min(x)`, `*w = w.max(x)`, `s.worst = s.worst.max(x)` |
| chained accumulator | `worst = worst.max(a).max(b)` |
| call-form accumulator | `worst = f64::max(worst, err)` |
| fold with a float reducer | `.fold(0.0, f64::max)`, `.fold(f64::INFINITY, f64::min)`, `f32::…` |
| fold with a closure reducer | `.fold(0.0, \|m, e\| m.max(e.abs()))`, `\|m, e\| f64::max(m, e)` |
| reduce | `.reduce(f64::max)`, `.reduce(\|a, b\| a.min(b))` |

Expressions inside macro calls are linted too, `assert!(…)` included. The defect
is most often a fold *inside* an assertion.

Not flagged, because the value cannot be `NaN`: any shape where every operand is
visibly an integer. That means an integer literal, `.len()`, `.count()`, an
`as usize` style cast, `usize::MAX`, or a local declared with an integer type or
integer literal. A `fold` whose initial value is an integer literal also counts.

## Escape hatches

1. **Assert the operand is finite.** Put an assert-family macro (`assert!`,
   `assert_eq!`, `debug_assert!`, `prop_assert!`, …) in the same function. Its
   arguments must call `is_finite` and name an identifier of the folded operand:

   ```rust
   for (got, want) in rows {
       assert!(got.is_finite(), "row {got} is not finite");
       worst = worst.max((got - want).abs());
   }
   // or, for a fold over a collection:
   assert!(errs.iter().all(|e| e.is_finite()));
   let worst = errs.iter().cloned().fold(0.0, f64::max);
   ```

   Asserting the **accumulator** is not a guard. `max` has already thrown the
   `NaN` away, so `assert!(worst.is_finite())` after the loop still passes.

2. **`// nan-ok: <reason>`** on any line of the flagged expression, or alone on
   the line directly above it. The reason is required, because a bare
   `// nan-ok:` excuses nothing:

   ```rust
   hi = hi.max(t); // nan-ok: t is a dataset TIME, validated finite by the reader
   ```

   Use it when `NaN` genuinely cannot reach the operand, or when the fold is
   not what the test asserts on.

Prefer (1). It turns the `NaN` into a red test, which is the point. Use (2) when
there is nothing to assert.

## The baseline

The sites that predate the rule (307, in 238 functions, when it landed) are
recorded per function in
[`baseline.txt`](baseline.txt). The ratchet works like `docs/.lint-baseline`:

- a finding in a function with no entry fails;
- a function whose count grew fails, and every finding in it is printed, since
  the new one cannot be told apart from the old ones;
- an entry whose function now has fewer findings fails as **stale**, so a
  paydown has to lower the line in the same PR.

After paying sites down, run `cargo run -p patterns-lint -- --update-baseline`.
It only lowers or removes entries. Do not add lines to make a red check green:
fix the fold or use an escape hatch. `--record-baseline` rewrites the file to
the current findings with growth included. It exists for the PR that introduces
a new rule, and the growth shows in that diff.
