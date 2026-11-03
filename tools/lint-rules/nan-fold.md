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
A local counts as an integer only where *that* declaration is in scope. A
`let worst = 0usize` inside a block says nothing about a float `worst` before
the block, after it, or in a sibling block, and a later float `let`, closure
parameter or loop pattern of the same name hides it.

## Escape hatches

1. **Assert the folded value is finite.** Put an `assert!` (or
   `prop_assert!`) *before* the fold whose condition
   *guarantees* the value the fold reads. Only positive conjuncts count: in
   `assert!(a.is_finite() && b.is_finite())` both are guaranteed, while
   `!x.is_finite()`, `x.is_finite() || y` and `assert_eq!(x.is_finite(), …)`
   guarantee nothing. `debug_assert!` doesn't count either. The nightly
   `ci-test` and `ci-fast` profiles inherit `release`, where it compiles to
   nothing, so the `NaN` it claims to catch is swallowed in exactly those runs.

   - **Where it holds.** The assert has to *dominate* the fold. It must be an
     earlier statement of the same block, or of a block that encloses the fold.
     `if check { assert!(e.is_finite()); }` guards nothing after the `if`, and
     an assert after the fold guards nothing. The guarantee is dropped when the
     value might be written: an assignment, `+=`, `&mut e`, a call of a method
     that might mutate it, such as `xs.push(…)`, or passing it as-is to a
     function or such a method (`overwrite(e)`), since it may itself be a
     `&mut`. `f(*e)`, `f(&e)` and `f64::…(e)` calls are not writes. An assert
     that itself writes the value, as in `assert!(e.is_finite() && refresh(&mut e))`,
     guards nothing. The write counts from where it
     is evaluated, so `if refresh(&mut e) { w = w.max(e) }` is not guarded, and
     neither is a fold in the body of a `for` or `match` whose iterator or
     scrutinee writes the value. It also lapses inside a loop body or a closure
     body that writes the value, because the next iteration or call runs after
     the write. It doesn't reach code where the name is rebound: a `let e`, a
     closure parameter `|e|`, or a `for e` / `match` / `if let` pattern.
   - **What it is about.** Only a *stable* expression can be asserted: locals,
     fields, indexing, literals, casts, arithmetic, pure float methods
     (`abs`, `sqrt`, `exp`, …), element-wise iterator adapters and closures
     built from those. A call is checked once and folded again, so
     `assert!(next().is_finite()); w = w.max(next())` is not guarded. Bind the
     value to a local, assert the local, and fold that.
   - **What it proves.** An accumulator operand is guarded when it is the
     asserted expression itself, or is built from asserted values and literals
     so that no `NaN` can come out. Finite inputs can still overflow to
     infinity, and two infinities can make a `NaN` (`inf - inf`, `inf * 0.0`,
     `inf / inf`). So the lint tracks which intermediates might be infinite.
     `(a - b).abs()`, `a * b + 1.0` and `(a - b).abs() / (1.0 + b.abs())` pass.
     `(a * b) * 0.0` and `a * b - a * b` don't. Division needs a divisor that is
     provably nonzero: a nonzero literal, `c + x.abs()` or `x.max(c)` with
     `c > 0`, because `0.0 / 0.0` is `NaN`. Anything else, such as a call,
     `sqrt`, `ln` or an unasserted field, is unguarded.
   - A `.fold`/`.reduce` is guarded by `C.all(|e| e.is_finite())` or
     `C.all(f64::is_finite)` over the **same** iterator chain `C` it folds.
     Element-preserving `iter`/`into_iter`/`cloned`/`copied` are ignored, but a
     `map` has to appear on both sides.

   ```rust
   for (got, want) in rows {
       assert!(got.is_finite() && want.is_finite(), "row {got} vs {want}");
       worst = worst.max((got - want).abs());
   }
   // or, for a fold over a collection:
   assert!(errs.iter().all(|e| e.is_finite()));
   let worst = errs.iter().cloned().fold(0.0, f64::max);
   ```

   Asserting only **one side** of a difference is not a guard: with
   `assert!(got.is_finite())`, a `NaN` `want` still vanishes into `max`.
   Asserting the **accumulator** is not a guard either. `max` has already
   thrown the `NaN` away, so `assert!(worst.is_finite())` after the loop still
   passes. If the other side is a committed reference you trust, say so with
   (2).

2. **`// nan-ok: <reason>`** on any line of the flagged expression, or alone on
   the line directly above it. It has to be a real `//` comment, so the same
   text inside a string literal doesn't count. The marker has to be exactly
   `nan-ok:` with the colon, and the reason is required. A bare `// nan-ok:` or
   `// nan-ok reason` excuses nothing:

   ```rust
   hi = hi.max(t); // nan-ok: t is a dataset TIME, validated finite by the reader
   ```

   Use it when `NaN` genuinely cannot reach the operand, or when the fold is
   not what the test asserts on.

Prefer (1). It turns the `NaN` into a red test, which is the point. Use (2) when
there is nothing to assert.

## The baseline

The sites that predate the rule (448, in 324 functions, when it landed) are
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
