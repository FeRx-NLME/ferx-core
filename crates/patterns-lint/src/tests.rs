//! Rule fixtures for `nan-fold`. Every fixture is a string, so this file — which
//! the corpus check also scans, being test code — never matches itself.

use crate::lint_source;

fn findings(src: &str) -> Vec<crate::Finding> {
    lint_source("tests/fixture.rs", src, true).unwrap()
}

fn lines(src: &str) -> Vec<usize> {
    findings(src).iter().map(|f| f.line).collect()
}

/// Wrap a body in a test fn.
fn t(body: &str) -> String {
    format!("#[test]\nfn t() {{\n{body}\n}}\n")
}

#[test]
fn flags_every_accumulator_shape() {
    for body in [
        "let mut worst = 0.0; for e in errs { worst = worst.max((e - 1.0).abs()); }",
        "let mut lo = f64::INFINITY; for e in errs { lo = lo.min(e); }",
        "let mut w = 0.0; w = w.max(a).max(b);",
        "let mut w = 0.0; w = f64::max(w, a);",
        "let mut w = 0.0; w = f64::min(a, w);",
        "let w = &mut s.worst; *w = w.max(a);",
        "s.worst = s.worst.max(a);",
        "w[0] = (w[0]).max(a);",
    ] {
        assert_eq!(findings(&t(body)).len(), 1, "not flagged: {body}");
    }
}

#[test]
fn flags_every_fold_and_reduce_shape() {
    for body in [
        "let m = errs.iter().cloned().fold(0.0, f64::max);",
        "let m = errs.iter().copied().fold(f64::INFINITY, f64::min);",
        "let m = errs.iter().fold(0.0_f32, f32::max);",
        "let m = errs.iter().fold(0.0, |m, &e| m.max(e));",
        "let m = errs.iter().fold(0.0, |m, e| { m.max(*e) });",
        "let m = errs.iter().fold(0.0, |m, e| f64::max(m, *e));",
        "let m = errs.iter().copied().reduce(f64::max);",
        "let m = errs.into_iter().reduce(|a, b| a.min(b));",
    ] {
        assert_eq!(findings(&t(body)).len(), 1, "not flagged: {body}");
    }
}

/// The defect lives inside assertions, which `syn` leaves as tokens.
#[test]
fn looks_inside_assert_and_format_macros() {
    let src = t("assert!(errs.iter().cloned().fold(0.0, f64::max) < 1e-6, \"{}\", 1);");
    assert_eq!(lines(&src), vec![3]);
    let src = t("println!(\"{}\", errs.iter().cloned().fold(0.0, f64::max));");
    assert_eq!(lines(&src), vec![3]);
    // A statement-bodied macro.
    let src = t("my_macro! { let mut w = 0.0; w = w.max(e); }");
    assert_eq!(lines(&src), vec![3]);
}

#[test]
fn does_not_flag_things_that_are_not_nan_swallowing_folds() {
    for body in [
        // Not an accumulator: a different place on each side.
        "let a = 0.0; let b = a.max(c);",
        "x = y.max(z);",
        // A NaN-propagating comparison, not max/min.
        "let m = errs.iter().fold(0.0, |m, e| if *e > m { *e } else { m });",
        "let m = errs.iter().fold(0.0, |m, e| m + e);",
        "let m = errs.iter().fold(0.0, std::cmp::max);",
        // Integers cannot be NaN.
        "let mut n = 0usize; for s in v { n = n.max(s.len()); }",
        "let mut n: u32 = 0; for s in v { n = n.max(s.k); }",
        "let mut n = 0; for s in v { n = n.max(s.k); }",
        "let mut w = 0.0; for s in v { w = w.max(s.k as usize); }",
        "let m = v.iter().fold(0, |m, s| m.max(s.len()));",
        "let mut n = usize::MAX; for s in v { n = n.min(s.k); }",
    ] {
        assert!(findings(&t(body)).is_empty(), "flagged: {body}");
    }
}

#[test]
fn an_is_finite_assert_on_the_operand_is_an_escape_hatch() {
    let guarded = t(
        "let mut w = 0.0;\nfor e in errs {\nassert!(e.is_finite(), \"row {e}\");\nw = w.max(e);\n}",
    );
    assert!(findings(&guarded).is_empty());
    // The whole-collection form, for a fold over the collection.
    let guarded = t(
        "assert!(errs.iter().all(|e| e.is_finite()));\nlet m = errs.iter().cloned().fold(0.0, f64::max);",
    );
    assert!(findings(&guarded).is_empty());
    let guarded = t("assert!(errs.iter().copied().all(f64::is_finite));\nlet m = errs.iter().cloned().fold(0.0, f64::max);");
    assert!(findings(&guarded).is_empty());
    // An operand composed of asserted values by NaN-free arithmetic.
    for body in [
        "assert!(a.is_finite() && b.is_finite()); w = w.max((a - b).abs());",
        "assert!(r.got.is_finite() && r.want.is_finite()); w = w.max((r.got - r.want).abs());",
        "assert!(g.is_finite() && fd.is_finite()); w = w.max((g - fd).abs() / (1.0 + fd.abs()));",
        "assert!(g.is_finite() && fd.is_finite()); w = w.max((g - fd).abs() / fd.abs().max(1e-12));",
        "assert!((got - want).is_finite()); w = w.max((got - want).abs());",
        "assert!(f64::is_finite(rel)); w = w.max(rel);",
        "assert!((e.is_finite())); w = w.max(-e * 2.0);",
        "assert!(a.is_finite() && b.is_finite()); w = w.max(a).max(b);",
        "assert!(errs.iter().all(|&e| e.is_finite())); let m = errs.iter().copied().fold(0.0, f64::max);",
        "assert!(xs.iter().map(|r| r.v).all(|v| v.is_finite())); let m = xs.iter().map(|r| r.v).fold(0.0, f64::max);",
    ] {
        assert!(findings(&t(body)).is_empty(), "flagged although guarded: {body}");
    }

    // Review round 1 (#2634): the guard must cover the VALUE that is folded, not
    // merely share a name with it, and must be asserted with positive polarity.
    for body in [
        // Only one side of the difference is asserted: `r.want = NaN` still vanishes.
        "for r in rows { assert!(r.got.is_finite()); w = w.max((r.got - r.want).abs()); }",
        "assert!(ferx.is_finite()); w = w.max((ferx - reference).abs());",
        // Negated: the assert guarantees the operand is NOT finite.
        "assert!(!err.is_finite()); w = w.max(err);",
        "assert!(!(err.is_finite())); w = w.max(err);",
        // A disjunction guarantees neither side.
        "assert!(err.is_finite() || other); w = w.max(err);",
        // `is_finite` buried in a non-conjunct expression.
        "assert!(err.is_finite() == flag); w = w.max(err);",
        // `assert_eq!(x.is_finite(), false)` is not an assert-on-the-condition.
        "assert_eq!(err.is_finite(), false); w = w.max(err);",
        // Division by a value that may be zero: 0/0 is NaN from finite inputs.
        "assert!(a.is_finite() && b.is_finite()); w = w.max(a / b);",
        // `sqrt` can make NaN from a finite negative.
        "assert!(a.is_finite()); w = w.max(a.sqrt());",
        // A chain with one operand unguarded.
        "assert!(a.is_finite()); w = w.max(a).max(b);",
        // The collection guard must check the same elements the fold reads.
        "assert!(xs.iter().all(|e| e.is_finite())); let m = xs.iter().map(|r| r.v).fold(0.0, f64::max);",
        "assert!(ys.iter().all(|e| e.is_finite())); let m = xs.iter().cloned().fold(0.0, f64::max);",
        // A predicate that checks something other than its parameter.
        "assert!(xs.iter().all(|e| other.is_finite())); let m = xs.iter().cloned().fold(0.0, f64::max);",
        "assert!(xs.iter().all(|e| !e.is_finite())); let m = xs.iter().cloned().fold(0.0, f64::max);",
        // `any` is not `all`.
        "assert!(xs.iter().any(|e| e.is_finite())); let m = xs.iter().cloned().fold(0.0, f64::max);",
    ] {
        assert_eq!(findings(&t(body)).len(), 1, "not flagged although unguarded: {body}");
    }

    // The assert must name the operand: asserting the ACCUMULATOR finite is no
    // guard at all — `max` already threw the NaN away.
    let wrong = t("let mut w = 0.0;\nfor e in errs { w = w.max(e); }\nassert!(w.is_finite());");
    assert_eq!(findings(&wrong).len(), 1);
    // …and must call `is_finite`: a plain assert naming the operand is not one.
    let wrong = t("for e in errs { assert!(e > 0.0); w = w.max(e); }");
    assert_eq!(findings(&wrong).len(), 1);
    // …and must be in the same function.
    let wrong = "#[test] fn a() { assert!(e.is_finite()); }\n#[test] fn b() { w = w.max(e); }\n";
    assert_eq!(findings(wrong).len(), 1);
}

#[test]
fn a_nan_ok_comment_with_a_reason_is_an_escape_hatch() {
    let ok = t("w = w.max(e); // nan-ok: e is a count of rows, never NaN");
    assert!(findings(&ok).is_empty());
    let above = t("// nan-ok: integer-valued f64\nw = w.max(e);");
    assert!(findings(&above).is_empty());
    // On any line of a wrapped expression.
    let wrapped = t("let m = errs\n.iter()\n.cloned() // nan-ok: errs asserted finite upstream\n.fold(0.0, f64::max);");
    assert!(findings(&wrapped).is_empty());

    // The reason is required, and the finding says so.
    let bare = findings(&t("w = w.max(e); // nan-ok:"));
    assert_eq!(bare.len(), 1);
    assert!(
        bare[0].message.contains("has no reason"),
        "{}",
        bare[0].message
    );
    // Two lines above is not "the line above".
    let far = t("// nan-ok: too far away\nlet x = 1;\nw = w.max(e);");
    assert_eq!(findings(&far).len(), 1);
    // A non-comment line above does not count even if it contains the text.
    let code_above = t("let s = \"// nan-ok: in a string\";\nw = w.max(e);");
    assert_eq!(findings(&code_above).len(), 1);
}

/// "Done when": a PR adding an unguarded fold goes red with a message naming
/// the rule and BOTH escape hatches.
#[test]
fn the_message_names_the_rule_and_both_escape_hatches() {
    let f = &findings(&t("let mut w = 0.0;\nw = w.max(e);"))[0];
    let shown = f.to_string();
    assert!(
        shown.starts_with("tests/fixture.rs:4: nan-fold: "),
        "{shown}"
    );
    assert!(shown.contains(".is_finite()"), "{shown}");
    assert!(shown.contains("// nan-ok: <reason>"), "{shown}");
    assert!(shown.contains("tools/lint-rules/nan-fold.md"), "{shown}");
    assert!(shown.contains("w = w.max(e);"), "the snippet: {shown}");
    assert_eq!(f.key, "tests/fixture.rs::t");
}

#[test]
fn only_test_code_is_linted_in_a_production_file() {
    let src = "\
fn prod(v: &[f64]) -> f64 { let mut w = 0.0; for e in v { w = w.max(*e); } w }
#[cfg(test)]
mod tests {
    fn helper(v: &[f64]) -> f64 { v.iter().cloned().fold(0.0, f64::max) }
    #[test]
    fn t() { let mut w = 0.0; w = w.max(1.5 * x); }
}
#[cfg(any(test, feature = \"nn\"))]
fn shared() { let mut w = 0.0; w = w.max(x); }
#[cfg(all(test, feature = \"nn\"))]
fn gated() { let mut w = 0.0; w = w.max(x); }
#[test]
fn top() { let mut w = 0.0; w = w.max(x); }
impl Foo {
    fn m(&self) { let mut w = 0.0; w = w.max(x); }
    #[cfg(test)]
    fn tm(&self) { let mut w = 0.0; w = w.max(x); }
}
";
    let got = lint_source("src/prod.rs", src, false).unwrap();
    let keys: Vec<&str> = got.iter().map(|f| f.key.as_str()).collect();
    assert_eq!(
        keys,
        vec![
            "src/prod.rs::tests::helper",
            "src/prod.rs::tests::t",
            "src/prod.rs::gated",
            "src/prod.rs::top",
            "src/prod.rs::impl Foo::tm",
        ]
    );
    // The same text as a whole-file test flags all of it.
    assert_eq!(lint_source("tests/x.rs", src, true).unwrap().len(), 8);
}

/// Review round 2 (#2634): a fact holds only where its assert DOMINATES the
/// fold, and only until the value is written or rebound.
#[test]
fn a_finite_assert_guards_only_the_code_it_dominates() {
    for body in [
        // Straight-line, and inside the loop body before the fold.
        "assert!(err.is_finite()); w = w.max(err);",
        "for e in errs { assert!(e.is_finite()); w = w.max(e); }",
        // An outer fact reaches into nested blocks it dominates.
        "assert!(err.is_finite()); if c { w = w.max(err); }",
        "assert!(err.is_finite()); for _ in 0..3 { w = w.max(err); }",
        // Asserted again after the write.
        "assert!(e.is_finite()); e = f(); assert!(e.is_finite()); w = w.max(e);",
        // A write to something else does not touch the fact.
        "assert!(e.is_finite()); other = 1.0; w = w.max(e);",
        "assert!(xs.iter().all(|e| e.is_finite())); let n = xs.len(); let m = xs.iter().cloned().fold(0.0, f64::max);",
    ] {
        assert!(findings(&t(body)).is_empty(), "flagged although guarded: {body}");
    }
    for body in [
        // The reviewer's case: the assert is conditional.
        "if check { assert!(err.is_finite()); } w = w.max(err);",
        "match k { 0 => assert!(err.is_finite()), _ => {} } w = w.max(err);",
        "{ assert!(err.is_finite()); } w = w.max(err);",
        // After, not before.
        "w = w.max(err); assert!(err.is_finite());",
        // Written between the assert and the fold.
        "assert!(e.is_finite()); e = f(); w = w.max(e);",
        "assert!(e.is_finite()); e += g; w = w.max(e);",
        "assert!(r.got.is_finite()); r.got = f(); w = w.max(r.got);",
        "assert!(e.is_finite()); poke(&mut e); w = w.max(e);",
        "assert!(e.is_finite()); if c { e = f(); } w = w.max(e);",
        "assert!(xs.iter().all(|e| e.is_finite())); xs.push(f64::NAN); let m = xs.iter().cloned().fold(0.0, f64::max);",
        // A loop body that writes the value: the second iteration is unguarded.
        "assert!(e.is_finite()); for _ in 0..3 { w = w.max(e); e = f(); }",
        "assert!(e.is_finite()); while c { w = w.max(e); e = f(); }",
        "assert!(e.is_finite()); loop { w = w.max(e); e = f(); }",
        // Rebound by a pattern: the fold reads a DIFFERENT `e`.
        "assert!(e.is_finite()); let e = f(); w = w.max(e);",
        "assert!(e.is_finite()); for e in errs { w = w.max(e); }",
        "assert!(e.is_finite()); errs.iter().for_each(|e| { w = w.max(*e); });",
        "assert!(e.is_finite()); if let Some(e) = o { w = w.max(e); }",
        "assert!(e.is_finite()); match o { Some(e) => { w = w.max(e); } None => {} }",
    ] {
        assert_eq!(findings(&t(body)).len(), 1, "not flagged although unguarded: {body}");
    }
    // A shadow is scoped: after the inner block the outer `e` and its fact return.
    let ok = t("assert!(e.is_finite()); { let e = f(); x = e; } w = w.max(e);");
    assert!(findings(&ok).is_empty());
}

/// Review round 2 (#2634): finite inputs do not make every combination NaN-free.
/// Two possibly-infinite intermediates can meet in `inf - inf`, `inf * 0.0` or
/// `inf / inf`.
#[test]
fn overflow_to_infinity_is_tracked_through_the_arithmetic() {
    let g = "assert!(a.is_finite() && b.is_finite());";
    for expr in [
        "(a * b) * 0.0",
        "a * b - a * b",
        "(a + b) - (a + b)",
        "(a * b) / (a * b + 1.0)",
        "(a - b).abs() / (1e300 + b.abs())",
        "a * b * a",
        // `a * 0.5` is finite, but `… - b` can still overflow before the `* 0.0`.
        "(a * 0.5 - b) * 0.0",
    ] {
        let src = t(&format!("{g} w = w.max({expr});"));
        assert_eq!(findings(&src).len(), 1, "not flagged: {expr}");
    }
    for expr in [
        // One possibly-infinite value: inf ± finite = inf, inf * 2.0 = inf.
        "a * b",
        "a * b + 1.0",
        "(a * b) * 2.0",
        // A small addend cannot overflow, so `1.0 + |b|` is finite and nonzero.
        "(a - b).abs() / (1.0 + b.abs())",
        "(a - b).abs() / b.abs().max(1e-12)",
        // `|x| <= 1` factors cannot grow a value.
        "(a * 0.5) * 0.0",
        "a / 2.0",
    ] {
        let src = t(&format!("{g} w = w.max({expr});"));
        assert!(
            findings(&src).is_empty(),
            "flagged although NaN-free: {expr}"
        );
    }
}

/// Review round 2 (#2634): `nan-ok` must be a real comment with the exact marker.
#[test]
fn nan_ok_is_lexed_not_searched() {
    // Inside a string literal on the line.
    for body in [
        "w = w.max(e); assert!(w < 1.0, \"// nan-ok: not a comment\");",
        "w = w.max(e); let s = r#\"// nan-ok: raw string\"#;",
        "w = w.max(e); let s = r\"// nan-ok: raw\";",
        "w = w.max(e); let c = '\"'; let s = \"// nan-ok: after a char\";",
        "w = w.max(e); /* // nan-ok: in a block comment */",
    ] {
        assert_eq!(
            findings(&t(body)).len(),
            1,
            "excused by a non-comment: {body}"
        );
    }
    // A string that spans lines does not leak comment state.
    let multi = t("let s = \"line one\n// nan-ok: inside the string\";\nw = w.max(e);");
    assert_eq!(findings(&multi).len(), 1);
    // Without the colon.
    let f = findings(&t("w = w.max(e); // nan-ok e is fine"));
    assert_eq!(f.len(), 1);
    assert!(
        f[0].message.contains("lacks the `nan-ok:` colon"),
        "{}",
        f[0].message
    );
    // A real comment after a string still counts; so does a lifetime nearby.
    let ok = t("w = w.max(e); let s = \"x\"; // nan-ok: e is an integer count");
    assert!(findings(&ok).is_empty());
    let ok = "#[test]\nfn t<'a>(e: &'a f64) {\nw = w.max(*e); // nan-ok: reason\n}\n";
    assert!(findings(ok).is_empty());
}

/// Review round 3 (#2634).
#[test]
fn debug_assert_is_not_a_guard_because_release_profiles_compile_it_out() {
    let src = t("debug_assert!(err.is_finite()); w = w.max(err);");
    assert_eq!(findings(&src).len(), 1);
    let src = t("debug_assert!(xs.iter().all(|e| e.is_finite())); let m = xs.iter().cloned().fold(0.0, f64::max);");
    assert_eq!(findings(&src).len(), 1);
    // `assert!` and `prop_assert!` always run.
    assert!(findings(&t("assert!(err.is_finite()); w = w.max(err);")).is_empty());
    assert!(findings(&t("prop_assert!(err.is_finite()); w = w.max(err);")).is_empty());
}

#[test]
fn passing_a_local_bare_to_a_call_may_write_it() {
    for body in [
        // `e: &mut f64` handed on as-is: the callee may write NaN through it.
        "assert!(e.is_finite()); overwrite(e); w = w.max(*e);",
        "assert!(s.x.is_finite()); refill(s.x); w = w.max(s.x);",
        "assert!(e.is_finite()); solver.step(e); w = w.max(*e);",
        "assert!(xs.iter().all(|e| e.is_finite())); fill(xs); let m = xs.iter().cloned().fold(0.0, f64::max);",
    ] {
        assert_eq!(findings(&t(body)).len(), 1, "not flagged although possibly written: {body}");
    }
    for body in [
        // A copy or a shared borrow cannot write the local.
        "assert!(e.is_finite()); log(*e); w = w.max(*e);",
        "assert!(e.is_finite()); show(&e); w = w.max(e);",
        // Arguments of read-only methods are not writes either.
        "assert!(e.is_finite() && a.is_finite()); let b = a.max(e); w = w.max(e);",
    ] {
        assert!(
            findings(&t(body)).is_empty(),
            "flagged although not written: {body}"
        );
    }
}

#[test]
fn integer_locals_are_lexically_scoped() {
    // The reviewer's case: an inner integer `worst` after the float fold.
    let src =
        t("let mut worst = 0.0; worst = worst.max(err); { let worst = 0usize; let _ = worst; }");
    assert_eq!(findings(&src).len(), 1);
    // …and before it, in a sibling block.
    let src =
        t("{ let worst = 0usize; let _ = worst; } let mut worst = 0.0; worst = worst.max(err);");
    assert_eq!(findings(&src).len(), 1);
    // A float rebinding of an integer name.
    let src = t("let mut n = 0usize; let mut n = 0.0; n = n.max(x);");
    assert_eq!(findings(&src).len(), 1);
    // A pattern rebinding hides the outer integer.
    let src = t("let n = 0usize; for n in floats { w = w.max(n); }");
    assert_eq!(findings(&src).len(), 1);
    let src = t("let mut n = 0usize; xs.iter().for_each(|n| { m = m.max(*n); });");
    assert_eq!(findings(&src).len(), 1);
    // The integer IS exempt where it is in scope, including nested blocks.
    let src = t("let mut worst = 0.0; worst = worst.max(e) /* nan-ok: unrelated */; { let mut k = 0usize; k = k.max(v.len()); }");
    assert_eq!(findings(&src).len(), 1, "only the float fold: {src}");
    let src = t("let mut k = 0usize; if c { k = k.max(x); }");
    assert!(findings(&src).is_empty());
    let src = t("let mut k = 0usize; { let mut j = 0.0; } k = k.max(x);");
    assert!(findings(&src).is_empty());
}
