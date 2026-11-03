//! Rule `nan-fold`: a `max`/`min` fold in test code that would swallow a `NaN`.
//!
//! `f64::max` and `f64::min` return the *other* operand when one is `NaN`
//! (`0.0f64.max(NaN) == 0.0`, `f64::INFINITY.min(NaN) == inf`), so
//!
//! ```text
//! worst = worst.max((got - want).abs());
//! assert!(worst < tol);
//! ```
//!
//! passes on the strength of the rows that worked: a solver regression that
//! returns `NaN` — the likeliest way to break the thing an anchor anchors —
//! goes green. `pktte_tdep_nonmem_anchor` had five such accumulators (AGENTS.md,
//! "A green test is not evidence that it can fail"). The full rule text, with
//! both escape hatches, is `tools/lint-rules/nan-fold.md`.
//!
//! Shapes flagged (in test code only — see [`crate::scope`]):
//!
//! - accumulators: `x = x.max(e)`, `x = x.min(e)`, chains of them
//!   (`x = x.max(a).max(b)`), and `x = f64::max(x, e)`;
//! - `.fold(init, f64::max)` / `f64::min` / `f32::…`, and a closure reducer whose
//!   body is a `max`/`min` (`.fold(0.0, |m, e| m.max(e))`);
//! - `.reduce(f64::max)` and the same closure forms.
//!
//! Not flagged, because `NaN` cannot occur: a fold whose every operand is
//! visibly an integer (an integer literal, `.len()`, `.count()`, an `as usize`
//! cast, or a local declared with an integer type or literal).
//!
//! Escape hatches:
//!
//! 1. an `assert!` / `prop_assert!` (not `debug_assert!`, which release-derived
//!    test profiles compile out) that DOMINATES the fold
//!    (an earlier statement of the fold's block or an enclosing one, with the
//!    value not written or rebound in between — see [`Scope`]) and whose
//!    condition guarantees the folded value finite. Only positive conjuncts
//!    count (`a && b`; never under `!` or `||`): `X.is_finite()` or
//!    `f64::is_finite(X)` guarantees `X`. An accumulator operand is guarded when
//!    it is such an `X`, or is built from guarded values and literals so that no
//!    NaN can arise even after overflow — see [`Class`]. A fold is guarded by
//!    `C.all(|e| e.is_finite())` / `C.all(f64::is_finite)` over the same
//!    iterator chain `C` it folds (`iter`/`cloned`/`copied` aside);
//! 2. `// nan-ok: <reason>` — a real `//` comment (lexed, so not text inside a
//!    string), with the exact `nan-ok:` marker and a non-empty reason — on any
//!    line of the flagged expression, or alone on the line above it.

use std::collections::BTreeSet;

use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{Expr, Token};

use crate::scope::attrs_mark_test;
use crate::Finding;

pub const RULE: &str = "nan-fold";

/// The text every finding ends with: the rule's id and both ways out.
pub const ESCAPES: &str =
    "f64::max/min discard NaN, so a NaN operand leaves the accumulator at the \
finite rows' value and the assertion passes green. Either assert the folded value \
itself `.is_finite()` before the fold, in the same block or an enclosing one \
(e.g. `assert!(err.is_finite())` before \
`w = w.max(err)`; for `(a - b).abs()` assert both `a` and `b`; for a `.fold` over \
`xs`, `assert!(xs.iter().all(|e| e.is_finite()))`), or put `// nan-ok: <reason>` \
on the line. See tools/lint-rules/nan-fold.md.";

const INT_TYPES: &[&str] = &[
    "usize", "isize", "u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128",
];

const INT_METHODS: &[&str] = &[
    "len",
    "count",
    "count_ones",
    "count_zeros",
    "leading_zeros",
    "trailing_zeros",
    "nrows",
    "ncols",
];

/// Lint one file's AST.
pub fn lint_file(rel: &str, text: &str, ast: &syn::File, whole_test: bool) -> Vec<Finding> {
    let mut l = Linter {
        rel,
        lines: text.lines().collect(),
        comments: line_comments(text),
        test_depth: usize::from(whole_test),
        path: Vec::new(),
        ctx: Vec::new(),
        out: Vec::new(),
    };
    l.visit_file(ast);
    l.out
}

/// A fact an `assert!` established: `key` is finite (or, for a chain, every
/// element is), stated in terms of the locals in `roots`.
#[derive(Clone)]
struct Fact {
    chain: bool,
    key: String,
    roots: BTreeSet<String>,
}

/// One lexical scope's worth of guard state.
///
/// A fact lives in the scope of the block whose statement asserted it, so it is
/// visible only to code that statement DOMINATES: the later statements of that
/// block and everything nested in them. `if c { assert!(e.is_finite()); }`
/// therefore guards nothing after the `if`. `masked` names locals that are
/// rebound in this scope (a `let`, a closure parameter, a `for`/`match`/`if let`
/// pattern) or that a loop body mutates, so facts from OUTER scopes about them
/// do not apply here.
///
/// `bound` names the locals this scope binds (by `let` or a pattern), and
/// `ints` the subset visibly declared as integers. An integer exemption is a
/// lookup through these, innermost first, so `{ let worst = 0usize; }` makes
/// `worst` an integer inside that block only — never the float `worst` outside
/// it, before it or after it (review 3).
#[derive(Default)]
struct Scope {
    facts: Vec<Fact>,
    masked: BTreeSet<String>,
    bound: BTreeSet<String>,
    ints: BTreeSet<String>,
}

/// What the escape-hatch and integer checks need to know about the enclosing fn.
#[derive(Default)]
struct FnCtx {
    /// The guard scopes, outermost first; see [`Scope`].
    scopes: Vec<Scope>,
}

impl FnCtx {
    fn has(&self, chain: bool, key: &str) -> bool {
        let mut blocked: BTreeSet<&String> = BTreeSet::new();
        for scope in self.scopes.iter().rev() {
            if scope.facts.iter().any(|f| {
                f.chain == chain && f.key == key && f.roots.iter().all(|r| !blocked.contains(r))
            }) {
                return true;
            }
            blocked.extend(scope.masked.iter());
        }
        false
    }

    /// A local was written: no fact about it survives, in any scope.
    fn invalidate(&mut self, roots: &BTreeSet<String>) {
        for scope in &mut self.scopes {
            scope.facts.retain(|f| f.roots.is_disjoint(roots));
        }
    }

    /// A local was rebound by a `let` in the innermost scope; `int` when the
    /// `let` visibly declares an integer.
    fn shadow(&mut self, names: BTreeSet<String>, int: bool) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.facts.retain(|f| f.roots.is_disjoint(&names));
            for n in &names {
                scope.ints.remove(n);
                if int {
                    scope.ints.insert(n.clone());
                }
            }
            scope.bound.extend(names.iter().cloned());
            scope.masked.extend(names);
        }
    }

    /// Is the local `name`, as resolved at this point, an integer? The
    /// innermost scope that binds it decides; a name no scope binds (a fn
    /// parameter, a static) is not known to be one.
    fn is_int_local(&self, name: &str) -> bool {
        for scope in self.scopes.iter().rev() {
            if scope.bound.contains(name) {
                return scope.ints.contains(name);
            }
        }
        false
    }
}

fn type_is_int(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(p) => p
            .path
            .get_ident()
            .is_some_and(|i| INT_TYPES.contains(&i.to_string().as_str())),
        syn::Type::Reference(r) => type_is_int(&r.elem),
        _ => false,
    }
}

/// Is `e` visibly an integer, so it can never be `NaN`?
fn expr_is_int(e: &Expr, int_locals: &dyn Fn(&str) -> bool) -> bool {
    match e {
        Expr::Lit(l) => matches!(l.lit, syn::Lit::Int(_)),
        Expr::Paren(p) => expr_is_int(&p.expr, int_locals),
        Expr::Reference(r) => expr_is_int(&r.expr, int_locals),
        Expr::Unary(u) => expr_is_int(&u.expr, int_locals),
        Expr::Cast(c) => type_is_int(&c.ty),
        Expr::MethodCall(m) => {
            let method = m.method.to_string();
            INT_METHODS.contains(&method.as_str())
                || (matches!(
                    method.as_str(),
                    "max" | "min" | "saturating_sub" | "saturating_add"
                ) && expr_is_int(&m.receiver, int_locals))
        }
        Expr::Path(p) => {
            if let Some(i) = p.path.get_ident() {
                return int_locals(&i.to_string());
            }
            // `usize::MAX`, `i64::MIN`.
            p.path
                .segments
                .first()
                .is_some_and(|s| INT_TYPES.contains(&s.ident.to_string().as_str()))
        }
        Expr::Binary(b) => expr_is_int(&b.left, int_locals) && expr_is_int(&b.right, int_locals),
        _ => false,
    }
}

/// The facts an `assert!`-shaped statement establishes, if it is one.
///
/// `assert!` / `prop_assert!` only: their first argument is a boolean the test
/// fails on. NOT `debug_assert!`: the nightly `ci-test` / `ci-fast` profiles
/// inherit `release`, where it compiles to nothing, so the NaN it claims to
/// catch is swallowed exactly there (review 3). `assert_eq!(x.is_finite(), false)` and friends
/// guarantee nothing, so they do not count.
fn assert_facts(m: &syn::Macro) -> Vec<Fact> {
    let name = m
        .path
        .segments
        .last()
        .map(|s| s.ident.to_string())
        .unwrap_or_default();
    let mut out = Vec::new();
    if matches!(name.as_str(), "assert" | "prop_assert") {
        if let Ok(args) = m.parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated) {
            if let Some(cond) = args.first() {
                collect_finite(cond, &mut out);
            }
        }
    }
    out
}

/// Walk an assert condition and record what it guarantees finite.
///
/// Only POSITIVE conjuncts guarantee anything: `a && b` guarantees both, a
/// parenthesised term is itself, and `!x.is_finite()`, `x.is_finite() || y`,
/// or an `is_finite` buried in any other expression guarantee nothing.
fn collect_finite(cond: &Expr, out: &mut Vec<Fact>) {
    let value = |e: &Expr| Fact {
        chain: false,
        key: value_key(e),
        roots: roots_of(e),
    };
    match cond {
        Expr::Paren(p) => collect_finite(&p.expr, out),
        Expr::Binary(b) if matches!(b.op, syn::BinOp::And(_)) => {
            collect_finite(&b.left, out);
            collect_finite(&b.right, out);
        }
        Expr::MethodCall(m) if m.method == "is_finite" && m.args.is_empty() => {
            out.push(value(&m.receiver));
        }
        Expr::Call(c) if c.args.len() == 1 && is_float_assoc_path(&c.func, "is_finite") => {
            out.push(value(&c.args[0]));
        }
        Expr::MethodCall(m)
            if m.method == "all" && m.args.len() == 1 && predicate_is_finite(&m.args[0]) =>
        {
            out.push(Fact {
                chain: true,
                key: chain_key(&m.receiver),
                roots: roots_of(&m.receiver),
            });
        }
        _ => {}
    }
}

/// Every single-identifier path in `e` — the locals a fact about `e` depends on.
fn roots_of(e: &Expr) -> BTreeSet<String> {
    struct R(BTreeSet<String>);
    impl<'ast> Visit<'ast> for R {
        fn visit_expr_path(&mut self, p: &'ast syn::ExprPath) {
            if let Some(i) = p.path.get_ident() {
                self.0.insert(i.to_string());
            }
        }
    }
    let mut r = R(BTreeSet::new());
    r.visit_expr(e);
    r.0
}

/// Every identifier a pattern binds.
fn pat_bindings(p: &syn::Pat) -> BTreeSet<String> {
    struct B(BTreeSet<String>);
    impl<'ast> Visit<'ast> for B {
        fn visit_pat_ident(&mut self, i: &'ast syn::PatIdent) {
            self.0.insert(i.ident.to_string());
            visit::visit_pat_ident(self, i);
        }
    }
    let mut b = B(BTreeSet::new());
    b.visit_pat(p);
    b.0
}

/// The local a place expression writes through: `a` for `a`, `a.b`, `a[i]`,
/// `*a`, `(a)`. `None` for a temporary (`f().x`, `v.iter()`).
fn place_root(e: &Expr) -> Option<String> {
    match e {
        Expr::Path(p) => p.path.get_ident().map(|i| i.to_string()),
        Expr::Field(f) => place_root(&f.base),
        Expr::Index(i) => place_root(&i.expr),
        Expr::Paren(p) => place_root(&p.expr),
        Expr::Unary(u) if matches!(u.op, syn::UnOp::Deref(_)) => place_root(&u.expr),
        Expr::Reference(r) => place_root(&r.expr),
        _ => None,
    }
}

/// The local behind an argument that is a place passed as-is (`e`, `s.x`,
/// `v[i]`) — not `*e` or `&e`.
fn bare_place_root(e: &Expr) -> Option<String> {
    match e {
        Expr::Path(_) | Expr::Field(_) | Expr::Index(_) => place_root(e),
        Expr::Paren(p) => bare_place_root(&p.expr),
        _ => None,
    }
}

/// Methods that do not mutate a place they are called on directly. Any other
/// method called on a local counts as a write to it — conservative, since a
/// missed write would let a stale fact excuse a fold.
const READ_ONLY_METHODS: &[&str] = &[
    "abs",
    "all",
    "any",
    "as_ref",
    "as_slice",
    "clone",
    "cloned",
    "column",
    "contains",
    "copied",
    "exp",
    "expect",
    "first",
    "get",
    "is_empty",
    "is_finite",
    "is_nan",
    "iter",
    "keys",
    "last",
    "len",
    "ln",
    "max",
    "min",
    "ncols",
    "nrows",
    "powf",
    "powi",
    "row",
    "sqrt",
    "to_string",
    "to_vec",
    "unwrap",
    "values",
];

/// The locals `e` ITSELF writes, not counting its subexpressions; see
/// [`mutated_roots`] for the cases.
fn direct_writes(e: &Expr) -> Vec<String> {
    let mut out = Vec::new();
    match e {
        Expr::Assign(a) => out.extend(place_root(&a.left)),
        Expr::Binary(b)
            if matches!(
                b.op,
                syn::BinOp::AddAssign(_)
                    | syn::BinOp::SubAssign(_)
                    | syn::BinOp::MulAssign(_)
                    | syn::BinOp::DivAssign(_)
                    | syn::BinOp::RemAssign(_)
            ) =>
        {
            out.extend(place_root(&b.left));
        }
        Expr::Reference(r) if r.mutability.is_some() => out.extend(place_root(&r.expr)),
        Expr::MethodCall(m) if !READ_ONLY_METHODS.contains(&m.method.to_string().as_str()) => {
            out.extend(place_root(&m.receiver));
            out.extend(m.args.iter().filter_map(bare_place_root));
        }
        Expr::Call(c) => out.extend(c.args.iter().filter_map(bare_place_root)),
        _ => {}
    }
    out
}

/// Every local a statement (at any depth) writes: assignment and compound
/// assignment targets, `&mut x` borrows, the receiver of any method call not in
/// [`READ_ONLY_METHODS`], and any local passed BARE as an argument to a function
/// or to such a method — it may itself be a `&mut` (`overwrite(e)` with
/// `e: &mut f64`, review 3). `f(*e)` and `f(&e)` pass a copy or a shared borrow
/// and are not writes.
fn mutated_roots<'a>(visit_fn: impl FnOnce(&mut dyn Visit<'a>)) -> BTreeSet<String> {
    struct M(BTreeSet<String>);
    impl<'ast> Visit<'ast> for M {
        fn visit_expr(&mut self, e: &'ast Expr) {
            self.0.extend(direct_writes(e));
            visit::visit_expr(self, e);
        }
    }
    let mut m = M(BTreeSet::new());
    visit_fn(&mut m);
    m.0
}

/// `f64::is_finite`, or a closure `|e| e.is_finite()` (any of `|&e|`, `|e: &f64|`)
/// whose body checks exactly its own parameter.
fn predicate_is_finite(p: &Expr) -> bool {
    if is_float_assoc_path(p, "is_finite") {
        return true;
    }
    let Expr::Closure(c) = p else { return false };
    let [param] = c.inputs.iter().collect::<Vec<_>>()[..] else {
        return false;
    };
    let Some(name) = pat_ident(param) else {
        return false;
    };
    match strip_block(&c.body) {
        Expr::MethodCall(m) if m.method == "is_finite" && m.args.is_empty() => {
            value_key(&m.receiver) == name
        }
        Expr::Call(call)
            if call.args.len() == 1 && is_float_assoc_path(&call.func, "is_finite") =>
        {
            value_key(&call.args[0]) == name
        }
        _ => false,
    }
}

fn pat_ident(p: &syn::Pat) -> Option<String> {
    match p {
        syn::Pat::Ident(i) => Some(i.ident.to_string()),
        syn::Pat::Reference(r) => pat_ident(&r.pat),
        syn::Pat::Type(t) => pat_ident(&t.pat),
        _ => None,
    }
}

/// Does this `let` visibly declare an integer — `let n: usize = …`, or an
/// integer-valued initialiser (`0`, `v.len()`, `x as u32`, another int local)?
fn local_is_int(local: &syn::Local, int_locals: &dyn Fn(&str) -> bool) -> bool {
    let ty = match &local.pat {
        syn::Pat::Type(pt) if matches!(&*pt.pat, syn::Pat::Ident(_)) => Some(&*pt.ty),
        syn::Pat::Ident(_) => None,
        _ => return false,
    };
    ty.is_some_and(type_is_int)
        || local
            .init
            .as_ref()
            .is_some_and(|i| expr_is_int(&i.expr, int_locals))
}

/// Identifiers bound by `let` patterns in an `if`/`while` condition.
fn let_bindings(cond: &Expr) -> BTreeSet<String> {
    struct L(BTreeSet<String>);
    impl<'ast> Visit<'ast> for L {
        fn visit_expr_let(&mut self, l: &'ast syn::ExprLet) {
            self.0.extend(pat_bindings(&l.pat));
            visit::visit_expr_let(self, l);
        }
    }
    let mut l = L(BTreeSet::new());
    l.visit_expr(cond);
    l.0
}

fn is_float_assoc_path(e: &Expr, name: &str) -> bool {
    let Expr::Path(p) = e else { return false };
    let segs: Vec<String> = p
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    segs.len() == 2 && (segs[0] == "f64" || segs[0] == "f32") && segs[1] == name
}

/// The identity of a VALUE for the finite guard: [`place_key`] with `.abs()`
/// peeled too, since `x.abs()` is finite exactly when `x` is. Nothing else is
/// peeled — `(got - want).abs()` is guarded by `got`'s assert only if `want`
/// cannot be NaN, which is not something this lint can see.
fn value_key(e: &Expr) -> String {
    match e {
        Expr::MethodCall(m) if m.method == "abs" && m.args.is_empty() => value_key(&m.receiver),
        Expr::Paren(p) => value_key(&p.expr),
        Expr::Unary(u) if matches!(u.op, syn::UnOp::Deref(_)) => value_key(&u.expr),
        Expr::Reference(r) => value_key(&r.expr),
        other => place_key(other),
    }
}

/// What the assert facts in scope prove about an expression's value.
///
/// `Finite` and `NotNan` are both safe to fold — an infinity reaches the
/// accumulator and fails a `< tol` assertion instead of vanishing — but they
/// are tracked apart, because combining two values that may be infinite CAN
/// make a NaN: `inf - inf`, `inf * 0.0`, `inf / inf`. With `a`, `b` asserted
/// finite, `a * b` may overflow, so `(a * b) * 0.0` is `Unknown` (review 2).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Class {
    Finite,
    NotNan,
    Unknown,
}

/// A literal small enough that adding it to any finite `f64` cannot round to
/// infinity: the spacing of doubles at `f64::MAX` is ~2e292, so a sum moves past
/// `MAX` only if the addend exceeds half of that.
const SAFE_ADDEND: f64 = 1e280;

fn lit_value(e: &Expr) -> Option<f64> {
    match e {
        Expr::Lit(l) => match &l.lit {
            syn::Lit::Float(f) => f.base10_parse::<f64>().ok(),
            syn::Lit::Int(i) => i.base10_parse::<u64>().ok().map(|v| v as f64),
            _ => None,
        },
        Expr::Paren(p) => lit_value(&p.expr),
        Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => lit_value(&u.expr).map(|v| -v),
        _ => None,
    }
}

fn classify(e: &Expr, ctx: &FnCtx) -> Class {
    use Class::*;
    if ctx.has(false, &value_key(e)) {
        return Finite;
    }
    if let Some(v) = lit_value(e) {
        return if v.is_finite() { Finite } else { NotNan };
    }
    match e {
        Expr::Paren(p) => classify(&p.expr, ctx),
        Expr::Reference(r) => classify(&r.expr, ctx),
        Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_) | syn::UnOp::Deref(_)) => {
            classify(&u.expr, ctx)
        }
        Expr::MethodCall(m) => match (m.method.to_string().as_str(), m.args.len()) {
            ("abs", 0) => classify(&m.receiver, ctx),
            ("max" | "min", 1) => classify(&m.receiver, ctx).max(classify(&m.args[0], ctx)),
            _ => Unknown,
        },
        Expr::Binary(b) => {
            let (l, r) = (classify(&b.left, ctx), classify(&b.right, ctx));
            let (lv, rv) = (lit_value(&b.left), lit_value(&b.right));
            match b.op {
                syn::BinOp::Add(_) | syn::BinOp::Sub(_) => match (l, r) {
                    (Unknown, _) | (_, Unknown) | (NotNan, NotNan) => Unknown,
                    (Finite, Finite) => {
                        let small = |v: Option<f64>| v.is_some_and(|v| v.abs() <= SAFE_ADDEND);
                        if small(lv) || small(rv) {
                            Finite
                        } else {
                            NotNan
                        }
                    }
                    _ => NotNan,
                },
                syn::BinOp::Mul(_) => {
                    let unit = |v: Option<f64>| v.is_some_and(|v| v.abs() <= 1.0);
                    let nonzero = |v: Option<f64>| v.is_some_and(|v| v != 0.0 && v.is_finite());
                    match (l, r) {
                        (Unknown, _) | (_, Unknown) => Unknown,
                        (Finite, Finite) if unit(lv) || unit(rv) => Finite,
                        (Finite, Finite) => NotNan,
                        // inf * 0.0 is NaN: a possibly-infinite factor is safe
                        // only against a nonzero literal.
                        (NotNan, Finite) if nonzero(rv) => NotNan,
                        (Finite, NotNan) if nonzero(lv) => NotNan,
                        _ => Unknown,
                    }
                }
                syn::BinOp::Div(_) => match (l, nonzero_divisor(&b.right, ctx)) {
                    // finite / nonzero (even infinite) is never NaN.
                    (Finite, Some(_)) => NotNan,
                    // inf / finite-nonzero is inf; inf / inf is NaN.
                    (NotNan, Some(Finite)) => NotNan,
                    _ => Unknown,
                },
                _ => Unknown,
            }
        }
        _ => Unknown,
    }
}

/// `Some(class)` when `e` is provably nonzero (and not NaN): a nonzero literal,
/// `c + |x|` / `|x| + c` with `c > 0`, or `x.max(c)` with `c > 0`.
fn nonzero_divisor(e: &Expr, ctx: &FnCtx) -> Option<Class> {
    let positive = |x: &Expr| lit_value(x).is_some_and(|v| v > 0.0 && v.is_finite());
    if let Some(v) = lit_value(e) {
        return (v != 0.0 && v.is_finite()).then_some(Class::Finite);
    }
    let class = classify(e, ctx);
    if class == Class::Unknown {
        return None;
    }
    match e {
        Expr::Paren(p) => nonzero_divisor(&p.expr, ctx),
        Expr::Binary(b) if matches!(b.op, syn::BinOp::Add(_)) => {
            ((positive(&b.left) && is_abs(&b.right)) || (positive(&b.right) && is_abs(&b.left)))
                .then_some(class)
        }
        Expr::MethodCall(m) if m.method == "max" && m.args.len() == 1 => {
            positive(&m.args[0]).then_some(class)
        }
        _ => None,
    }
}

fn is_abs(e: &Expr) -> bool {
    match e {
        Expr::Paren(p) => is_abs(&p.expr),
        Expr::MethodCall(m) => m.method == "abs" && m.args.is_empty(),
        _ => false,
    }
}

/// The identity of an ITERATOR CHAIN for the collection guard: element-
/// preserving adapters (`iter`, `into_iter`, `cloned`, `copied`, `by_ref`) are
/// peeled from the end, so `errs.iter().all(…)` guards `errs.iter().cloned()
/// .fold(…)`. An adapter that CHANGES the elements (`map`, `zip`, …) is kept,
/// so the assert has to check the same mapped values the fold reads.
fn chain_key(e: &Expr) -> String {
    match e {
        Expr::MethodCall(m)
            if m.args.is_empty()
                && matches!(
                    m.method.to_string().as_str(),
                    "iter" | "into_iter" | "cloned" | "copied" | "by_ref"
                ) =>
        {
            chain_key(&m.receiver)
        }
        Expr::Paren(p) => chain_key(&p.expr),
        Expr::Reference(r) => chain_key(&r.expr),
        other => place_key(other),
    }
}

fn quote_into(e: &Expr, ts: &mut TokenStream) {
    e.to_tokens(ts);
}

/// `e` with parentheses, `*` and `&` peeled, as a token string — the identity
/// used to decide that `*m = m.max(v)` updates the receiver it reads.
fn place_key(e: &Expr) -> String {
    match e {
        Expr::Paren(p) => place_key(&p.expr),
        Expr::Unary(u) if matches!(u.op, syn::UnOp::Deref(_)) => place_key(&u.expr),
        Expr::Reference(r) => place_key(&r.expr),
        other => {
            let mut ts = TokenStream::new();
            quote_into(other, &mut ts);
            ts.to_string()
        }
    }
}

fn is_float_max_min_path(e: &Expr) -> bool {
    let Expr::Path(p) = e else { return false };
    let segs: Vec<String> = p
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    matches!(
        segs.iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
        ["f64" | "f32", "max" | "min"]
    )
}

fn is_max_min(name: &str) -> bool {
    name == "max" || name == "min"
}

fn strip_block(e: &Expr) -> &Expr {
    match e {
        Expr::Paren(p) => strip_block(&p.expr),
        Expr::Block(b) if b.block.stmts.len() == 1 => match &b.block.stmts[0] {
            syn::Stmt::Expr(inner, None) => strip_block(inner),
            _ => e,
        },
        _ => e,
    }
}

/// Is a `fold`/`reduce` reducer a NaN-swallowing `max`/`min`?
fn reducer_swallows_nan(r: &Expr) -> bool {
    if is_float_max_min_path(r) {
        return true;
    }
    let Expr::Closure(c) = r else { return false };
    match strip_block(&c.body) {
        Expr::MethodCall(m) => is_max_min(&m.method.to_string()),
        Expr::Call(call) => is_float_max_min_path(&call.func),
        _ => false,
    }
}

struct Linter<'a> {
    rel: &'a str,
    lines: Vec<&'a str>,
    comments: Vec<LineComment>,
    test_depth: usize,
    path: Vec<String>,
    ctx: Vec<FnCtx>,
    out: Vec<Finding>,
}

/// What a finite assert has to cover for a finding to be excused.
enum Guard<'e> {
    /// An accumulator's operands, each of which must be asserted finite.
    Values(Vec<&'e Expr>),
    /// A fold's receiver chain, which must be asserted all-finite.
    Chain(&'e Expr),
}

enum NanOk {
    Absent,
    Excused,
    MissingReason(usize),
    MissingColon(usize),
}

/// The `//` comment on one source line, found by lexing — so `"// nan-ok: x"`
/// inside a string literal is not a comment — and whether the line holds any
/// code outside comments.
#[derive(Default, Clone)]
struct LineComment {
    text: Option<String>,
    has_code: bool,
}

/// Lex `text` just far enough to find line comments: string, raw-string, byte-
/// string and char literals are skipped, as are (nested) block comments.
fn line_comments(text: &str) -> Vec<LineComment> {
    let c: Vec<char> = text.chars().collect();
    let mut out = vec![LineComment::default(); text.lines().count() + 1];
    let (mut i, mut line) = (0usize, 0usize);
    let ident = |ch: char| ch.is_alphanumeric() || ch == '_';
    let mark = |out: &mut Vec<LineComment>, line: usize| {
        if let Some(l) = out.get_mut(line) {
            l.has_code = true;
        }
    };
    while i < c.len() {
        let ch = c[i];
        let next = c.get(i + 1).copied();
        match ch {
            '\n' => {
                line += 1;
                i += 1;
            }
            '/' if next == Some('/') => {
                let start = i + 2;
                let mut j = start;
                while j < c.len() && c[j] != '\n' {
                    j += 1;
                }
                if let Some(l) = out.get_mut(line) {
                    l.text = Some(c[start..j].iter().collect());
                }
                i = j;
            }
            '/' if next == Some('*') => {
                let mut depth = 1;
                i += 2;
                while i < c.len() && depth > 0 {
                    if c[i] == '\n' {
                        line += 1;
                    } else if c[i] == '/' && c.get(i + 1) == Some(&'*') {
                        depth += 1;
                        i += 1;
                    } else if c[i] == '*' && c.get(i + 1) == Some(&'/') {
                        depth -= 1;
                        i += 1;
                    }
                    i += 1;
                }
            }
            'r' if (i == 0
                || !ident(c[i - 1])
                || (c[i - 1] == 'b' && (i < 2 || !ident(c[i - 2]))))
                && matches!(next, Some('"') | Some('#')) =>
            {
                // A raw string `r"…"` / `r#"…"#` (or `br…`); `r#ident` is not one.
                let mut j = i + 1;
                let mut hashes = 0;
                while c.get(j) == Some(&'#') {
                    hashes += 1;
                    j += 1;
                }
                if c.get(j) != Some(&'"') {
                    mark(&mut out, line);
                    i += 1;
                    continue;
                }
                mark(&mut out, line);
                j += 1;
                loop {
                    match c.get(j) {
                        None => break,
                        Some('\n') => line += 1,
                        Some('"') if (1..=hashes).all(|k| c.get(j + k) == Some(&'#')) => {
                            j += 1 + hashes;
                            break;
                        }
                        _ => {}
                    }
                    mark(&mut out, line);
                    j += 1;
                }
                i = j;
            }
            '"' => {
                mark(&mut out, line);
                i += 1;
                while i < c.len() && c[i] != '"' {
                    if c[i] == '\\' {
                        i += 1;
                    }
                    if c.get(i) == Some(&'\n') {
                        line += 1;
                    }
                    mark(&mut out, line);
                    i += 1;
                }
                i += 1;
            }
            '\'' => {
                mark(&mut out, line);
                // A char literal `'x'` / `'\n'`, or else a lifetime `'a`.
                if next == Some('\\') {
                    i += 2;
                    while i < c.len() && c[i] != '\'' {
                        i += 1;
                    }
                    i += 1;
                } else if c.get(i + 2) == Some(&'\'') {
                    i += 3;
                } else {
                    i += 1;
                }
            }
            ch if ch.is_whitespace() => i += 1,
            _ => {
                mark(&mut out, line);
                i += 1;
            }
        }
    }
    out
}

impl Linter<'_> {
    fn in_test(&self) -> bool {
        self.test_depth > 0
    }

    fn key(&self) -> String {
        if self.path.is_empty() {
            self.rel.to_string()
        } else {
            format!("{}::{}", self.rel, self.path.join("::"))
        }
    }

    /// `// nan-ok: <reason>` on a line of `[first, last]`, or alone on the line
    /// above. Only a real comment counts (see [`line_comments`]), and only with
    /// the exact `nan-ok:` marker.
    fn nan_ok(&self, first: usize, last: usize) -> NanOk {
        let comment = |n: usize| self.comments.get(n.wrapping_sub(1));
        let mut candidates: Vec<usize> = (first..=last).collect();
        if first > 1 && comment(first - 1).is_some_and(|c| c.text.is_some() && !c.has_code) {
            candidates.push(first - 1);
        }
        let mut problem = NanOk::Absent;
        for n in candidates {
            let Some(text) = comment(n).and_then(|c| c.text.as_deref()) else {
                continue;
            };
            // `/// nan-ok:` and `//! nan-ok:` too.
            let body = text.trim_start_matches(['/', '!']).trim_start();
            let Some(rest) = body.strip_prefix("nan-ok") else {
                continue;
            };
            match rest.strip_prefix(':') {
                Some(reason) if !reason.trim().is_empty() => return NanOk::Excused,
                Some(_) => problem = NanOk::MissingReason(n),
                None => problem = NanOk::MissingColon(n),
            }
        }
        problem
    }

    fn guarded(&self, guard: &Guard) -> bool {
        let Some(ctx) = self.ctx.last() else {
            return false;
        };
        match guard {
            // EVERY operand must be asserted finite itself: `w.max(a).max(b)`
            // with only `a` asserted still swallows a NaN `b`.
            Guard::Values(ops) => ops.iter().all(|e| classify(e, ctx) != Class::Unknown),
            Guard::Chain(c) => ctx.has(true, &chain_key(c)),
        }
    }

    /// Is `name` an integer local at this point of the walk?
    fn is_int_local(&self, name: &str) -> bool {
        self.ctx.last().is_some_and(|c| c.is_int_local(name))
    }

    fn report(&mut self, whole: &dyn Spanned, at_line: usize, shape: &str, guard: Guard) {
        if self.guarded(&guard) {
            return;
        }
        let span = whole.span();
        let (first, last) = (span.start().line, span.end().line);
        let note = match self.nan_ok(first, last) {
            NanOk::Excused => return,
            NanOk::Absent => String::new(),
            NanOk::MissingReason(n) => {
                format!(" (the `// nan-ok` on line {n} has no reason after the colon, so it excuses nothing)")
            }
            NanOk::MissingColon(n) => {
                format!(" (the `// nan-ok` on line {n} lacks the `nan-ok:` colon, so it excuses nothing)")
            }
        };
        let snippet = self
            .lines
            .get(at_line.saturating_sub(1))
            .map(|l| l.trim().to_string())
            .unwrap_or_default();
        self.out.push(Finding {
            rule: RULE,
            file: self.rel.to_string(),
            line: at_line,
            key: self.key(),
            message: format!("{shape} in test code{note}. {ESCAPES}"),
            snippet,
        });
    }

    fn check_assign(&mut self, a: &syn::ExprAssign) {
        let lhs = place_key(&a.left);
        let mut operands: Vec<&Expr> = Vec::new();
        let mut cur: &Expr = &a.right;
        let mut call_form = false;
        loop {
            match cur {
                Expr::Paren(p) => cur = &p.expr,
                Expr::MethodCall(m) if is_max_min(&m.method.to_string()) && m.args.len() == 1 => {
                    operands.push(&m.args[0]);
                    cur = &m.receiver;
                }
                Expr::Call(c) if is_float_max_min_path(&c.func) && c.args.len() == 2 => {
                    if place_key(&c.args[0]) == lhs {
                        operands.push(&c.args[1]);
                    } else if place_key(&c.args[1]) == lhs {
                        operands.push(&c.args[0]);
                    } else {
                        return;
                    }
                    call_form = true;
                    break;
                }
                _ => break,
            }
        }
        if operands.is_empty() || (!call_form && place_key(cur) != lhs) {
            return;
        }
        let ints = |n: &str| self.is_int_local(n);
        if expr_is_int(&a.left, &ints) || operands.iter().all(|o| expr_is_int(o, &ints)) {
            return;
        }
        let line = a.span().start().line;
        self.report(a, line, "`max`/`min` accumulator", Guard::Values(operands));
    }

    fn check_fold(&mut self, m: &syn::ExprMethodCall) {
        let method = m.method.to_string();
        let (reducer, init) = match (method.as_str(), m.args.len()) {
            ("fold", 2) => (&m.args[1], Some(&m.args[0])),
            ("reduce", 1) => (&m.args[0], None),
            _ => return,
        };
        if !reducer_swallows_nan(reducer) {
            return;
        }
        if init.is_some_and(|i| expr_is_int(i, &|n| self.is_int_local(n))) {
            return;
        }
        let line = m.method.span().start().line;
        let shape = format!("`.{method}(…)` with a `max`/`min` reducer");
        self.report(m, line, &shape, Guard::Chain(&m.receiver));
    }

    fn enter(&mut self, name: String, attrs: &[syn::Attribute]) -> bool {
        self.path.push(name);
        let t = attrs_mark_test(attrs);
        if t {
            self.test_depth += 1;
        }
        t
    }

    /// A scope that masks outer facts about `masked` and binds `bound` (with
    /// no type known, so not as integers).
    fn push_scope(&mut self, masked: BTreeSet<String>, bound: BTreeSet<String>) {
        if let Some(ctx) = self.ctx.last_mut() {
            ctx.scopes.push(Scope {
                masked,
                bound,
                ..Scope::default()
            });
        }
    }

    fn pop_scope(&mut self) {
        if let Some(ctx) = self.ctx.last_mut() {
            ctx.scopes.pop();
        }
    }

    fn after_stmt(&mut self, stmt: &syn::Stmt) {
        let Some(ctx) = self.ctx.last_mut() else {
            return;
        };
        ctx.invalidate(&mutated_roots(|v| v.visit_stmt(stmt)));
        let mac = match stmt {
            syn::Stmt::Local(l) => {
                let int = local_is_int(l, &|n| ctx.is_int_local(n));
                ctx.shadow(pat_bindings(&l.pat), int);
                None
            }
            syn::Stmt::Macro(m) => Some(&m.mac),
            syn::Stmt::Expr(Expr::Macro(m), _) => Some(&m.mac),
            _ => None,
        };
        if let (Some(mac), Some(scope)) = (mac, ctx.scopes.last_mut()) {
            scope.facts.extend(assert_facts(mac));
        }
    }

    fn leave(&mut self, was_test: bool) {
        self.path.pop();
        if was_test {
            self.test_depth -= 1;
        }
    }
}

impl<'ast> Visit<'ast> for Linter<'_> {
    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        let t = self.enter(i.ident.to_string(), &i.attrs);
        visit::visit_item_mod(self, i);
        self.leave(t);
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        let mut ts = TokenStream::new();
        i.self_ty.to_tokens(&mut ts);
        let name = format!("impl {}", ts.to_string().replace(' ', ""));
        let t = self.enter(name, &i.attrs);
        visit::visit_item_impl(self, i);
        self.leave(t);
    }

    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        let t = self.enter(i.sig.ident.to_string(), &i.attrs);
        self.ctx.push(FnCtx::default());
        visit::visit_item_fn(self, i);
        self.ctx.pop();
        self.leave(t);
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        let t = self.enter(i.sig.ident.to_string(), &i.attrs);
        self.ctx.push(FnCtx::default());
        visit::visit_impl_item_fn(self, i);
        self.ctx.pop();
        self.leave(t);
    }

    fn visit_trait_item_fn(&mut self, i: &'ast syn::TraitItemFn) {
        let t = self.enter(i.sig.ident.to_string(), &i.attrs);
        self.ctx.push(FnCtx::default());
        visit::visit_trait_item_fn(self, i);
        self.ctx.pop();
        self.leave(t);
    }

    /// Statements in order, each followed by its effect on the guard facts:
    /// writes invalidate, `let` shadows, and an assert adds what it proves —
    /// for the rest of THIS block only.
    fn visit_block(&mut self, b: &'ast syn::Block) {
        if !self.in_test() || self.ctx.is_empty() {
            return visit::visit_block(self, b);
        }
        self.push_scope(BTreeSet::new(), BTreeSet::new());
        for stmt in &b.stmts {
            self.visit_stmt(stmt);
            self.after_stmt(stmt);
        }
        self.pop_scope();
    }

    fn visit_expr_closure(&mut self, c: &'ast syn::ExprClosure) {
        let params: BTreeSet<String> = c.inputs.iter().flat_map(pat_bindings).collect();
        self.push_scope(params.clone(), params);
        visit::visit_expr_closure(self, c);
        self.pop_scope();
    }

    /// A loop body runs again after its own writes, so facts from outside about
    /// anything the body writes — or the loop pattern binds — do not hold in it.
    fn visit_expr_for_loop(&mut self, f: &'ast syn::ExprForLoop) {
        self.visit_expr(&f.expr);
        let bound = pat_bindings(&f.pat);
        let mut masked = bound.clone();
        masked.extend(mutated_roots(|v| v.visit_block(&f.body)));
        self.push_scope(masked, bound);
        self.visit_block(&f.body);
        self.pop_scope();
    }

    fn visit_expr_while(&mut self, w: &'ast syn::ExprWhile) {
        let bound = let_bindings(&w.cond);
        let mut masked = mutated_roots(|v| v.visit_block(&w.body));
        masked.extend(bound.iter().cloned());
        masked.extend(mutated_roots(|v| v.visit_expr(&w.cond)));
        self.push_scope(masked, bound);
        self.visit_expr(&w.cond);
        self.visit_block(&w.body);
        self.pop_scope();
    }

    fn visit_expr_loop(&mut self, l: &'ast syn::ExprLoop) {
        let masked = mutated_roots(|v| v.visit_block(&l.body));
        self.push_scope(masked, BTreeSet::new());
        self.visit_block(&l.body);
        self.pop_scope();
    }

    fn visit_expr_if(&mut self, i: &'ast syn::ExprIf) {
        let bound = let_bindings(&i.cond);
        self.push_scope(bound.clone(), bound);
        visit::visit_expr_if(self, i);
        self.pop_scope();
    }

    fn visit_arm(&mut self, a: &'ast syn::Arm) {
        let bound = pat_bindings(&a.pat);
        self.push_scope(bound.clone(), bound);
        visit::visit_arm(self, a);
        self.pop_scope();
    }

    fn visit_expr(&mut self, e: &'ast Expr) {
        if self.in_test() {
            match e {
                Expr::Assign(a) => self.check_assign(a),
                Expr::MethodCall(m) => self.check_fold(m),
                _ => {}
            }
        }
        visit::visit_expr(self, e);
        // A write takes effect where it is evaluated, not at the end of the
        // statement: `if refresh(&mut e) { w = w.max(e) }` must not see the
        // fact `refresh` may have broken (review 4). Children are visited
        // first, in syn's (left-to-right, i.e. evaluation) order, so the
        // invalidation lands before any later operand or body is checked.
        if self.in_test() {
            if let Some(ctx) = self.ctx.last_mut() {
                ctx.invalidate(&direct_writes(e).into_iter().collect());
            }
        }
    }

    /// `syn` leaves macro bodies as tokens, and the defect lives *inside*
    /// assertions — `assert!(errs.iter().fold(0.0, f64::max) < tol)` — so parse
    /// the body as comma-separated expressions (every assert and format macro),
    /// else as statements, and lint what comes out.
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        if self.in_test() {
            if let Ok(exprs) = m.parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated) {
                for e in &exprs {
                    self.visit_expr(e);
                }
            } else if let Ok(stmts) = m.parse_body_with(syn::Block::parse_within) {
                for s in &stmts {
                    self.visit_stmt(s);
                }
            }
        }
        visit::visit_macro(self, m);
    }
}
