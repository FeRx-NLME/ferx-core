//! Rule `recorded-value-read`: `.plain()` (the one value read left on a generic number,
//! `sens::num::NonRecRead::plain`) and `.probe_value()` (a documented non-decision) outside the arms where nothing is being recorded (#2786).
//!
//! `PkNum` has no `val()` / `seen_value()`: a `*_g<T: PkNum>` kernel cannot read a value at
//! all, so the compiler already rejects the reads the rule used to hunt. What it can still do
//! is call `.plain()`, which reads a value when `T` is a plain number and **fails the
//! recording** (falls back to finite differences) when `T` is the recording type and the
//! value depends on an input. This rule keeps that single escape confined to code that runs
//! only for plain types. Full text: `tools/lint-rules/recorded-value-read.md`.
//!
//! Scope: production code of `src/` (not tests; see [`crate::scope`]). A function is
//! *generic over a number type* when it, or the `impl`/`trait` around it, has a type
//! parameter bounded by `PkNum`, `VmNum` or `RecHooks` (inline, in a `where` clause, or as
//! `impl PkNum`), or when its signature or `impl` target names `Tr`. There, every
//! zero-argument `.plain()` call and every `NonRecRead::plain` / `X::plain` path is a finding.
//!
//! Not flagged:
//!
//! - an arm that provably runs only when `T::RECORDS_BRANCHES` is false:
//!   `if !T::RECORDS_BRANCHES { .. }`, the `else` of `if T::RECORDS_BRANCHES { .. }`, the
//!   right operand of `!T::RECORDS_BRANCHES && ..` / `T::RECORDS_BRANCHES || ..`, the
//!   statements after `if T::RECORDS_BRANCHES { return .. }`. `&&`, `||` and `!` are
//!   evaluated, so `if T::RECORDS_BRANCHES && x { .. } else { .. }` exempts neither arm;
//! - a `#[cfg(test)]` / `#[test]` item or a test file;
//! - a function whose doc comment has a line beginning `Non-recording: <reason>`;
//! - a line carrying `// value-read-ok: <reason>` (or alone on the line above), for a read
//!   that cannot affect recorded control flow.

use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{BinOp, Expr, Token, UnOp};

use crate::nan_fold::{line_comments, LineComment};
use crate::scope::attrs_mark_test;
use crate::Finding;

pub const RULE: &str = "recorded-value-read";

const READS: &[&str] = &["plain", "probe_value"];
const NUM_TRAITS: &[&str] = &["PkNum", "VmNum", "RecHooks", "NonRecRead"];

pub const ESCAPES: &str = "A recording (`Tr`) must never read a value: `.plain()` fails it on a dependent value and the \
subject falls back to finite differences (`.probe_value()` hands over the value it saw, unguarded: only for a computation whose result is checked not to depend on it). Decide through the branch API (`br_cmp`, `br_cmp_c`, \
`br_abs_lt`, `br_is_nan`, `select_if`), put the read behind `!T::RECORDS_BRANCHES`, or, if it \
cannot affect recorded control flow, put `// value-read-ok: <reason>` on the line. \
See tools/lint-rules/recorded-value-read.md.";

/// Lint one file's AST. Only `src/` production code is in scope.
pub fn lint_file(rel: &str, text: &str, ast: &syn::File, whole_test: bool) -> Vec<Finding> {
    if whole_test || !rel.starts_with("src/") {
        return Vec::new();
    }
    let mut l = Linter {
        rel,
        lines: text.lines().collect(),
        comments: line_comments(text),
        test_depth: 0,
        nonrec: 0,
        path: Vec::new(),
        generic: Vec::new(),
        aliases: Vec::new(),
        out: Vec::new(),
    };
    l.visit_file(ast);
    l.out
}

struct Linter<'a> {
    rel: &'a str,
    lines: Vec<&'a str>,
    comments: Vec<LineComment>,
    test_depth: usize,
    /// > 0 inside code that provably runs only when `RECORDS_BRANCHES` is false.
    nonrec: usize,
    path: Vec<String>,
    /// Is the enclosing item generic over a number type? Innermost last.
    generic: Vec<bool>,
    /// Locals bound to `T::RECORDS_BRANCHES`.
    aliases: Vec<String>,
    out: Vec<Finding>,
}

fn last_ident(p: &syn::Path) -> Option<String> {
    p.segments.last().map(|s| s.ident.to_string())
}

fn bounds_num(bounds: &Punctuated<syn::TypeParamBound, Token![+]>) -> bool {
    bounds.iter().any(|b| match b {
        syn::TypeParamBound::Trait(t) => {
            last_ident(&t.path).is_some_and(|n| NUM_TRAITS.contains(&n.as_str()))
        }
        _ => false,
    })
}

fn generics_num(g: &syn::Generics) -> bool {
    g.params
        .iter()
        .any(|p| matches!(p, syn::GenericParam::Type(t) if bounds_num(&t.bounds)))
        || g.where_clause.as_ref().is_some_and(|w| {
            w.predicates
                .iter()
                .any(|p| matches!(p, syn::WherePredicate::Type(t) if bounds_num(&t.bounds)))
        })
}

/// Does a signature / type mention `Tr`, or take `impl PkNum`?
#[derive(Default)]
struct SigScan(bool);
impl<'ast> Visit<'ast> for SigScan {
    fn visit_path(&mut self, p: &'ast syn::Path) {
        if p.segments.iter().any(|s| s.ident == "Tr") {
            self.0 = true;
        }
        visit::visit_path(self, p);
    }
    fn visit_type_impl_trait(&mut self, t: &'ast syn::TypeImplTrait) {
        if bounds_num(&t.bounds) {
            self.0 = true;
        }
        visit::visit_type_impl_trait(self, t);
    }
}

fn sig_num(sig: &syn::Signature) -> bool {
    let mut s = SigScan::default();
    s.visit_signature(sig);
    generics_num(&sig.generics) || s.0
}

/// `T::RECORDS_BRANCHES`, or a local bound to it (`let rec = T::RECORDS_BRANCHES;`).
fn is_records_branches(e: &Expr, aliases: &[String]) -> bool {
    matches!(e, Expr::Path(p)
        if p.path.segments.last().is_some_and(|s| s.ident == "RECORDS_BRANCHES")
            || (p.path.segments.len() == 1
                && aliases.iter().any(|a| p.path.is_ident(a))))
}

/// Does `cond == truth` imply `RECORDS_BRANCHES` is false?
fn nonrec_when(cond: &Expr, truth: bool, al: &[String]) -> bool {
    match cond {
        Expr::Paren(p) => nonrec_when(&p.expr, truth, al),
        Expr::Group(g) => nonrec_when(&g.expr, truth, al),
        Expr::Unary(u) if matches!(u.op, UnOp::Not(_)) => nonrec_when(&u.expr, !truth, al),
        e if is_records_branches(e, al) => !truth,
        Expr::Binary(b) if matches!(b.op, BinOp::And(_)) => {
            if truth {
                nonrec_when(&b.left, true, al) || nonrec_when(&b.right, true, al)
            } else {
                nonrec_when(&b.left, false, al) && nonrec_when(&b.right, false, al)
            }
        }
        Expr::Binary(b) if matches!(b.op, BinOp::Or(_)) => {
            if truth {
                nonrec_when(&b.left, true, al) && nonrec_when(&b.right, true, al)
            } else {
                nonrec_when(&b.left, false, al) || nonrec_when(&b.right, false, al)
            }
        }
        _ => false,
    }
}

fn block_diverges(b: &syn::Block) -> bool {
    matches!(
        b.stmts.last(),
        Some(syn::Stmt::Expr(
            Expr::Return(_) | Expr::Break(_) | Expr::Continue(_),
            _
        ))
    )
}

fn doc_non_recording(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| {
        if !a.path().is_ident("doc") {
            return false;
        }
        let syn::Meta::NameValue(nv) = &a.meta else {
            return false;
        };
        let syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Str(s),
            ..
        }) = &nv.value
        else {
            return false;
        };
        s.value()
            .trim()
            .strip_prefix("Non-recording:")
            .is_some_and(|r| !r.trim().is_empty())
    })
}

impl Linter<'_> {
    fn in_generic(&self) -> bool {
        self.generic.last().copied().unwrap_or(false)
    }

    fn key(&self) -> String {
        if self.path.is_empty() {
            self.rel.to_string()
        } else {
            format!("{}::{}", self.rel, self.path.join("::"))
        }
    }

    /// `// value-read-ok: <reason>` on a line of `[first, last]` or alone on the one above.
    fn marked(&self, first: usize, last: usize) -> bool {
        let comment = |n: usize| self.comments.get(n.wrapping_sub(1));
        let mut cands: Vec<usize> = (first..=last).collect();
        if first > 1 && comment(first - 1).is_some_and(|c| c.text.is_some() && !c.has_code) {
            cands.push(first - 1);
        }
        cands.into_iter().any(|n| {
            comment(n)
                .and_then(|c| c.text.as_deref())
                .and_then(|t| {
                    t.trim_start_matches(['/', '!'])
                        .trim_start()
                        .strip_prefix("value-read-ok:")
                })
                .is_some_and(|r| !r.trim().is_empty())
        })
    }

    fn report(&mut self, first: usize, last: usize, what: &str) {
        // A function *named* like the read is the primitive itself (`DualT::probe_value`
        // delegating to its base): its callers are what the rule lints.
        let is_primitive = self
            .path
            .last()
            .is_some_and(|f| READS.contains(&f.as_str()));
        if self.test_depth > 0 || self.nonrec > 0 || is_primitive || self.marked(first, last) {
            return;
        }
        let snippet = self
            .lines
            .get(first.saturating_sub(1))
            .map(|l| l.trim().to_string())
            .unwrap_or_default();
        self.out.push(Finding {
            rule: RULE,
            file: self.rel.to_string(),
            line: first,
            key: self.key(),
            message: format!("{what} outside a `!T::RECORDS_BRANCHES` arm. {ESCAPES}"),
            snippet,
        });
    }

    /// Run `f` with `nonrec` raised when `exempt`.
    fn with_nonrec(&mut self, exempt: bool, f: impl FnOnce(&mut Self)) {
        self.nonrec += usize::from(exempt);
        f(self);
        self.nonrec -= usize::from(exempt);
    }

    fn enter(&mut self, name: String, attrs: &[syn::Attribute], generic: bool) -> (bool, bool) {
        self.path.push(name);
        let t = attrs_mark_test(attrs);
        self.test_depth += usize::from(t);
        let nr = doc_non_recording(attrs);
        self.nonrec += usize::from(nr);
        self.generic.push(generic);
        (t, nr)
    }

    fn leave(&mut self, (t, nr): (bool, bool)) {
        self.generic.pop();
        self.path.pop();
        self.test_depth -= usize::from(t);
        self.nonrec -= usize::from(nr);
    }
}

impl<'ast> Visit<'ast> for Linter<'_> {
    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        let t = self.enter(i.ident.to_string(), &i.attrs, false);
        visit::visit_item_mod(self, i);
        self.leave(t);
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        let mut ts = TokenStream::new();
        i.self_ty.to_tokens(&mut ts);
        let mut sc = SigScan::default();
        sc.visit_type(&i.self_ty);
        if let Some((_, p, _)) = &i.trait_ {
            sc.visit_path(p);
        }
        let name = format!("impl {}", ts.to_string().replace(' ', ""));
        let t = self.enter(name, &i.attrs, generics_num(&i.generics) || sc.0);
        visit::visit_item_impl(self, i);
        self.leave(t);
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        let own = NUM_TRAITS.contains(&i.ident.to_string().as_str());
        let t = self.enter(
            format!("trait {}", i.ident),
            &i.attrs,
            own || generics_num(&i.generics),
        );
        visit::visit_item_trait(self, i);
        self.leave(t);
    }

    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        // A free fn does not inherit generics from where it is written.
        let t = self.enter(i.sig.ident.to_string(), &i.attrs, sig_num(&i.sig));
        visit::visit_item_fn(self, i);
        self.leave(t);
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        let g = self.in_generic() || sig_num(&i.sig);
        let t = self.enter(i.sig.ident.to_string(), &i.attrs, g);
        visit::visit_impl_item_fn(self, i);
        self.leave(t);
    }

    fn visit_trait_item_fn(&mut self, i: &'ast syn::TraitItemFn) {
        let g = self.in_generic() || sig_num(&i.sig);
        let t = self.enter(i.sig.ident.to_string(), &i.attrs, g);
        visit::visit_trait_item_fn(self, i);
        self.leave(t);
    }

    fn visit_block(&mut self, b: &'ast syn::Block) {
        let mut raised = 0;
        for stmt in &b.stmts {
            if let syn::Stmt::Local(l) = stmt {
                if let (syn::Pat::Ident(pi), Some(init)) = (&l.pat, &l.init) {
                    if is_records_branches(&init.expr, &self.aliases) {
                        self.aliases.push(pi.ident.to_string());
                    }
                }
            }
            self.visit_stmt(stmt);
            // `if T::RECORDS_BRANCHES { return .. }` leaves only non-recording code after it.
            if let syn::Stmt::Expr(Expr::If(i), _) = stmt {
                if i.else_branch.is_none()
                    && block_diverges(&i.then_branch)
                    && nonrec_when(&i.cond, false, &self.aliases)
                {
                    self.nonrec += 1;
                    raised += 1;
                }
            }
        }
        self.nonrec -= raised;
    }

    fn visit_expr_if(&mut self, i: &'ast syn::ExprIf) {
        self.visit_expr(&i.cond);
        self.with_nonrec(nonrec_when(&i.cond, true, &self.aliases), |s| {
            s.visit_block(&i.then_branch)
        });
        if let Some((_, e)) = &i.else_branch {
            self.with_nonrec(nonrec_when(&i.cond, false, &self.aliases), |s| {
                s.visit_expr(e)
            });
        }
    }

    fn visit_expr_binary(&mut self, b: &'ast syn::ExprBinary) {
        self.visit_expr(&b.left);
        let exempt = match b.op {
            BinOp::And(_) => nonrec_when(&b.left, true, &self.aliases),
            BinOp::Or(_) => nonrec_when(&b.left, false, &self.aliases),
            _ => false,
        };
        self.with_nonrec(exempt, |s| s.visit_expr(&b.right));
    }

    fn visit_expr_method_call(&mut self, m: &'ast syn::ExprMethodCall) {
        if m.args.is_empty() && self.in_generic() && READS.contains(&m.method.to_string().as_str())
        {
            let first = m.method.span().start().line;
            let last = m.span().end().line.max(first);
            self.report(
                first,
                last,
                &format!("`.{}()` on a number-type value", m.method),
            );
        }
        visit::visit_expr_method_call(self, m);
    }

    fn visit_expr_path(&mut self, p: &'ast syn::ExprPath) {
        let segs = &p.path.segments;
        if segs.len() >= 2 {
            let last = segs.last().map(|s| s.ident.to_string()).unwrap_or_default();
            if READS.contains(&last.as_str()) && self.in_generic() {
                let line = p.span().start().line;
                self.report(
                    line,
                    p.span().end().line.max(line),
                    &format!("`{last}` path"),
                );
            }
        }
        visit::visit_expr_path(self, p);
    }

    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        if let Ok(exprs) = m.parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated) {
            for e in &exprs {
                self.visit_expr(e);
            }
        } else if let Ok(stmts) = m.parse_body_with(syn::Block::parse_within) {
            for s in &stmts {
                self.visit_stmt(s);
            }
        }
        visit::visit_macro(self, m);
    }
}
