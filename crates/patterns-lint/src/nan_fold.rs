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
//! 1. an assert-family macro (`assert!`, `assert_eq!`, `debug_assert!`,
//!    `prop_assert!`, …) in the same function whose arguments call `is_finite`
//!    and name an identifier of the folded operand — `assert!(err.is_finite())`
//!    before `worst = worst.max(err)`, or
//!    `assert!(errs.iter().all(|e| e.is_finite()))` before folding `errs`;
//! 2. `// nan-ok: <reason>` on any line of the flagged expression, or alone on
//!    the line above it. The reason is required.

use std::collections::BTreeSet;

use proc_macro2::{TokenStream, TokenTree};
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
finite rows' value and the assertion passes green. Either assert the operand \
`.is_finite()` in this function (e.g. `assert!(err.is_finite())` before the fold), \
or put `// nan-ok: <reason>` on the line. See tools/lint-rules/nan-fold.md.";

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
        test_depth: usize::from(whole_test),
        path: Vec::new(),
        ctx: Vec::new(),
        out: Vec::new(),
    };
    l.visit_file(ast);
    l.out
}

/// What the escape-hatch and integer checks need to know about the enclosing fn.
#[derive(Default)]
struct FnCtx {
    /// One identifier set per assert-family macro that calls `is_finite`.
    finite_asserts: Vec<BTreeSet<String>>,
    /// Locals visibly declared as integers.
    int_locals: BTreeSet<String>,
}

impl FnCtx {
    fn scan(block: &syn::Block) -> Self {
        struct Scan(FnCtx);
        impl<'ast> Visit<'ast> for Scan {
            fn visit_macro(&mut self, m: &'ast syn::Macro) {
                let name = m
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                if name.contains("assert") {
                    let idents = idents_of(&m.tokens, false);
                    if idents.contains("is_finite") {
                        self.0.finite_asserts.push(idents);
                    }
                }
                visit::visit_macro(self, m);
            }
            fn visit_local(&mut self, local: &'ast syn::Local) {
                let (name, ty) = match &local.pat {
                    syn::Pat::Ident(p) => (Some(p.ident.to_string()), None),
                    syn::Pat::Type(pt) => match &*pt.pat {
                        syn::Pat::Ident(p) => (Some(p.ident.to_string()), Some(&*pt.ty)),
                        _ => (None, None),
                    },
                    _ => (None, None),
                };
                if let Some(name) = name {
                    let int_ty = ty.is_some_and(type_is_int);
                    let int_init = local
                        .init
                        .as_ref()
                        .is_some_and(|i| expr_is_int(&i.expr, &BTreeSet::new()));
                    if int_ty || int_init {
                        self.0.int_locals.insert(name);
                    }
                }
                visit::visit_local(self, local);
            }
        }
        let mut s = Scan(FnCtx::default());
        s.visit_block(block);
        s.0
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
fn expr_is_int(e: &Expr, int_locals: &BTreeSet<String>) -> bool {
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
                return int_locals.contains(&i.to_string());
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

/// Identifiers in a token stream. With `skip_members`, an identifier directly
/// after a `.` (a method or field name) is not counted: for the folded operand
/// `(r.got - r.want).abs()` the identifiers that can carry a guard are `r`, not
/// `abs`.
fn idents_of(tokens: &TokenStream, skip_members: bool) -> BTreeSet<String> {
    fn walk(tokens: TokenStream, skip_members: bool, out: &mut BTreeSet<String>) {
        let mut after_dot = false;
        for tt in tokens {
            match tt {
                TokenTree::Ident(i) => {
                    if !(skip_members && after_dot) {
                        out.insert(i.to_string());
                    }
                    after_dot = false;
                }
                TokenTree::Punct(p) => after_dot = p.as_char() == '.',
                TokenTree::Group(g) => {
                    walk(g.stream(), skip_members, out);
                    after_dot = false;
                }
                TokenTree::Literal(_) => after_dot = false,
            }
        }
    }
    let mut out = BTreeSet::new();
    walk(tokens.clone(), skip_members, &mut out);
    out
}

fn expr_idents(e: &Expr) -> BTreeSet<String> {
    let mut ts = TokenStream::new();
    quote_into(e, &mut ts);
    idents_of(&ts, true)
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
    test_depth: usize,
    path: Vec<String>,
    ctx: Vec<FnCtx>,
    out: Vec<Finding>,
}

enum NanOk {
    Absent,
    Excused,
    MissingReason(usize),
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

    /// `// nan-ok: <reason>` on a line of `[first, last]`, or alone on the line above.
    fn nan_ok(&self, first: usize, last: usize) -> NanOk {
        let comment_reason = |line: &str| -> Option<String> {
            let at = line.find("//")?;
            let rest = line[at + 2..].trim_start();
            let rest = rest.strip_prefix("nan-ok")?;
            let rest = rest.trim_start().strip_prefix(':').unwrap_or(rest);
            Some(rest.trim().to_string())
        };
        let mut candidates: Vec<usize> = (first..=last).collect();
        if first > 1 {
            let above = self.lines.get(first - 2).copied().unwrap_or_default();
            if above.trim_start().starts_with("//") {
                candidates.push(first - 1);
            }
        }
        let mut missing = None;
        for n in candidates {
            let Some(line) = self.lines.get(n - 1) else {
                continue;
            };
            match comment_reason(line) {
                Some(reason) if !reason.is_empty() => return NanOk::Excused,
                Some(_) => missing = Some(n),
                None => {}
            }
        }
        missing.map_or(NanOk::Absent, NanOk::MissingReason)
    }

    fn guarded(&self, operands: &[&Expr]) -> bool {
        let Some(ctx) = self.ctx.last() else {
            return false;
        };
        let wanted: BTreeSet<String> = operands.iter().flat_map(|e| expr_idents(e)).collect();
        ctx.finite_asserts
            .iter()
            .any(|set| set.iter().any(|i| wanted.contains(i)))
    }

    fn int_locals(&self) -> BTreeSet<String> {
        self.ctx
            .last()
            .map(|c| c.int_locals.clone())
            .unwrap_or_default()
    }

    fn report(&mut self, whole: &dyn Spanned, at_line: usize, shape: &str, operands: &[&Expr]) {
        if self.guarded(operands) {
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
        let ints = self.int_locals();
        if expr_is_int(&a.left, &ints) || operands.iter().all(|o| expr_is_int(o, &ints)) {
            return;
        }
        let line = a.span().start().line;
        self.report(a, line, "`max`/`min` accumulator", &operands);
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
        if init.is_some_and(|i| expr_is_int(i, &self.int_locals())) {
            return;
        }
        let line = m.method.span().start().line;
        let shape = format!("`.{method}(…)` with a `max`/`min` reducer");
        self.report(m, line, &shape, &[&m.receiver]);
    }

    fn enter(&mut self, name: String, attrs: &[syn::Attribute]) -> bool {
        self.path.push(name);
        let t = attrs_mark_test(attrs);
        if t {
            self.test_depth += 1;
        }
        t
    }

    /// Production fns are walked (a `#[cfg(test)]` item may nest inside one)
    /// but never reported on, so only a test fn pays for the guard scan.
    fn scan_if_test(&self, block: &syn::Block) -> FnCtx {
        if self.in_test() {
            FnCtx::scan(block)
        } else {
            FnCtx::default()
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
        let scanned = self.scan_if_test(&i.block);
        self.ctx.push(scanned);
        visit::visit_item_fn(self, i);
        self.ctx.pop();
        self.leave(t);
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        let t = self.enter(i.sig.ident.to_string(), &i.attrs);
        let scanned = self.scan_if_test(&i.block);
        self.ctx.push(scanned);
        visit::visit_impl_item_fn(self, i);
        self.ctx.pop();
        self.leave(t);
    }

    fn visit_trait_item_fn(&mut self, i: &'ast syn::TraitItemFn) {
        let t = self.enter(i.sig.ident.to_string(), &i.attrs);
        let scanned = i
            .default
            .as_ref()
            .map(|b| self.scan_if_test(b))
            .unwrap_or_default();
        self.ctx.push(scanned);
        visit::visit_trait_item_fn(self, i);
        self.ctx.pop();
        self.leave(t);
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
