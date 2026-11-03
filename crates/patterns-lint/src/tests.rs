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
    // A field operand is guarded through its base.
    let guarded =
        t("for r in rows { assert!(r.got.is_finite()); w = w.max((r.got - r.want).abs()); }");
    assert!(findings(&guarded).is_empty());

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
