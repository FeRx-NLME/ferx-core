//! Shared expression / condition tokenizer and recursive-descent parser.
//!
//! Used by [`crate::parser::model_parser`] (the `.ferx` DSL parser) and by
//! [`crate::parser::mrgsolve`] for any expression-shaped fragment whose
//! syntax matches ferx's: numeric literals, bare identifiers, the four
//! arithmetic ops, `^`, parenthesised subexpressions, single-arg function
//! calls, the inline `if (cond) a else b` form, and comparison/logical
//! operators inside conditions.
//!
//! Statement-level parsing (block bodies, `[odes]` `d/dt(...)`, multi-line
//! if/else) lives in `model_parser.rs` for the ferx DSL and in
//! `mrgsolve/cpp_stmt.rs` for the mrgsolve C++-flavoured bodies. Both reuse
//! `parse_expression` / `parse_condition` from here.

use crate::parser::ast::{BinOp, CmpOp, Condition, Expression};

// ── ParseCtx ────────────────────────────────────────────────────────────────

/// Context threaded through the recursive-descent parser so that every bare
/// identifier can be classified as Theta / Eta / Variable / Covariate without
/// relying on casing heuristics.
#[derive(Clone, Copy)]
pub(crate) struct ParseCtx<'a> {
    pub(crate) theta_names: &'a [String],
    pub(crate) eta_names: &'a [String],
    /// Names previously assigned in the surrounding block (e.g. earlier lines
    /// of [individual_parameters]). These resolve to `Variable`.
    pub(crate) defined_vars: &'a [String],
    /// When `true` (the usual case), an unknown identifier is a covariate.
    /// Set to `false` for the ODE RHS parser, where state names and individual
    /// parameters are injected into the `vars` map at eval time instead.
    pub(crate) fallback_covariate: bool,
}

impl<'a> ParseCtx<'a> {
    pub(crate) fn new(
        theta_names: &'a [String],
        eta_names: &'a [String],
        defined_vars: &'a [String],
    ) -> Self {
        Self {
            theta_names,
            eta_names,
            defined_vars,
            fallback_covariate: true,
        }
    }

    pub(crate) fn ode(defined_vars: &'a [String]) -> Self {
        const EMPTY: &[String] = &[];
        Self {
            theta_names: EMPTY,
            eta_names: EMPTY,
            defined_vars,
            fallback_covariate: false,
        }
    }
}

// ── Tokenizer ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Token {
    Number(f64),
    Ident(String),
    LParen,
    RParen,
    LBrace,
    RBrace,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    /// `=` — used by the statement parser as the assignment operator. The
    /// expression parser never accepts it, so `==` (`Token::EqEq`) is the
    /// correct choice for equality comparisons inside conditions.
    Eq,
    EqEq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    AndAnd,
    OrOr,
    Bang,
    Comma,
    /// Logical line separator. Consumed only by the statement parser, where
    /// it acts as an "end of statement" marker. Newlines inside `(...)` and
    /// `[...]` are stripped by `strip_newlines_in_groups` so users can break
    /// long expressions across lines.
    Newline,
}

pub(crate) fn tokenize(s: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\r' => i += 1,
            '\n' => {
                // Collapse adjacent newlines so the statement parser doesn't
                // see runs of empty lines as separate terminators.
                if !matches!(tokens.last(), Some(Token::Newline)) {
                    tokens.push(Token::Newline);
                }
                i += 1;
            }
            '#' => {
                // Line comment — skip to next newline. Block-level comments
                // were already stripped by extract_blocks, but inline `# ...`
                // tails inside a multi-line if-block (which we re-join with
                // newlines) need to be tolerated here too.
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '{' => {
                tokens.push(Token::LBrace);
                i += 1;
            }
            '}' => {
                tokens.push(Token::RBrace);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            '<' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Le);
                    i += 2;
                } else {
                    tokens.push(Token::Lt);
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Ge);
                    i += 2;
                } else {
                    tokens.push(Token::Gt);
                    i += 1;
                }
            }
            '=' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::EqEq);
                    i += 2;
                } else {
                    tokens.push(Token::Eq);
                    i += 1;
                }
            }
            '!' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Ne);
                    i += 2;
                } else {
                    tokens.push(Token::Bang);
                    i += 1;
                }
            }
            '&' => {
                if i + 1 < chars.len() && chars[i + 1] == '&' {
                    tokens.push(Token::AndAnd);
                    i += 2;
                } else {
                    return Err("Unexpected '&' (use '&&' for logical and)".to_string());
                }
            }
            '|' => {
                if i + 1 < chars.len() && chars[i + 1] == '|' {
                    tokens.push(Token::OrOr);
                    i += 2;
                } else {
                    return Err("Unexpected '|' (use '||' for logical or)".to_string());
                }
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                // Check if this is a negative number (after operator or at start)
                let is_unary = tokens.is_empty()
                    || matches!(
                        tokens.last(),
                        Some(
                            Token::LParen
                                | Token::LBrace
                                | Token::Plus
                                | Token::Minus
                                | Token::Star
                                | Token::Slash
                                | Token::Caret
                                | Token::Eq
                                | Token::EqEq
                                | Token::Ne
                                | Token::Lt
                                | Token::Le
                                | Token::Gt
                                | Token::Ge
                                | Token::AndAnd
                                | Token::OrOr
                                | Token::Bang
                                | Token::Comma
                                | Token::Newline
                        )
                    );
                if is_unary
                    && i + 1 < chars.len()
                    && (chars[i + 1].is_ascii_digit() || chars[i + 1] == '.')
                {
                    let start = i;
                    i += 1;
                    while i < chars.len()
                        && (chars[i].is_ascii_digit()
                            || chars[i] == '.'
                            || chars[i] == 'e'
                            || chars[i] == 'E')
                    {
                        i += 1;
                    }
                    let num_str: String = chars[start..i].iter().collect();
                    let num: f64 = num_str
                        .parse()
                        .map_err(|_| format!("Bad number: {}", num_str))?;
                    tokens.push(Token::Number(num));
                } else {
                    tokens.push(Token::Minus);
                    i += 1;
                }
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '^' => {
                tokens.push(Token::Caret);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'e'
                        || chars[i] == 'E')
                {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let num: f64 = num_str
                    .parse()
                    .map_err(|_| format!("Bad number: {}", num_str))?;
                tokens.push(Token::Number(num));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(ident));
            }
            other => return Err(format!("Unexpected character: {}", other)),
        }
    }

    Ok(tokens)
}

// ── Recursive descent parser ────────────────────────────────────────────────

/// Parse one expression (the standard entry point used by callers outside
/// this module). Equivalent to `parse_add_sub`, but stable as a public API.
#[allow(dead_code)] // consumed by the mrgsolve front-end (Phase 1)
pub(crate) fn parse_expression(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Expression, usize), String> {
    parse_add_sub(tokens, pos, ctx)
}

pub(crate) fn parse_add_sub(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Expression, usize), String> {
    let (mut left, mut pos) = parse_mul_div(tokens, pos, ctx)?;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Plus => {
                let (right, p) = parse_mul_div(tokens, pos + 1, ctx)?;
                left = Expression::BinOp(Box::new(left), BinOp::Add, Box::new(right));
                pos = p;
            }
            Token::Minus => {
                let (right, p) = parse_mul_div(tokens, pos + 1, ctx)?;
                left = Expression::BinOp(Box::new(left), BinOp::Sub, Box::new(right));
                pos = p;
            }
            _ => break,
        }
    }

    Ok((left, pos))
}

fn parse_mul_div(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Expression, usize), String> {
    let (mut left, mut pos) = parse_power(tokens, pos, ctx)?;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Star => {
                let (right, p) = parse_power(tokens, pos + 1, ctx)?;
                left = Expression::BinOp(Box::new(left), BinOp::Mul, Box::new(right));
                pos = p;
            }
            Token::Slash => {
                let (right, p) = parse_power(tokens, pos + 1, ctx)?;
                left = Expression::BinOp(Box::new(left), BinOp::Div, Box::new(right));
                pos = p;
            }
            _ => break,
        }
    }

    Ok((left, pos))
}

fn parse_power(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Expression, usize), String> {
    let (base, mut pos) = parse_atom(tokens, pos, ctx)?;

    if pos < tokens.len() && tokens[pos] == Token::Caret {
        let (exp, p) = parse_atom(tokens, pos + 1, ctx)?;
        pos = p;
        return Ok((Expression::Power(Box::new(base), Box::new(exp)), pos));
    }

    Ok((base, pos))
}

fn parse_atom(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Expression, usize), String> {
    if pos >= tokens.len() {
        return Err("Unexpected end of expression".to_string());
    }

    match &tokens[pos] {
        Token::Minus => {
            // Unary minus: -expr → 0 - expr
            let (expr, p) = parse_atom(tokens, pos + 1, ctx)?;
            Ok((
                Expression::BinOp(
                    Box::new(Expression::Literal(0.0)),
                    BinOp::Sub,
                    Box::new(expr),
                ),
                p,
            ))
        }
        Token::Number(n) => Ok((Expression::Literal(*n), pos + 1)),
        Token::LParen => {
            let (expr, p) = parse_add_sub(tokens, pos + 1, ctx)?;
            if p >= tokens.len() || tokens[p] != Token::RParen {
                return Err("Missing closing parenthesis".to_string());
            }
            Ok((expr, p + 1))
        }
        Token::Ident(name) => {
            // Inline conditional expression: `if (cond) then_expr else else_expr`
            // Used as a value (e.g. `CL = if (SEX == 1) TVCL * 1.2 else TVCL`).
            if name.eq_ignore_ascii_case("if") {
                let p = pos + 1;
                if p >= tokens.len() || tokens[p] != Token::LParen {
                    return Err("`if` must be followed by `(`".to_string());
                }
                let (cond, p) = parse_condition(tokens, p + 1, ctx)?;
                if p >= tokens.len() || tokens[p] != Token::RParen {
                    return Err("Missing closing `)` after if-condition".to_string());
                }
                let (then_expr, p) = parse_add_sub(tokens, p + 1, ctx)?;
                if p >= tokens.len() {
                    return Err("Inline `if` expression missing `else` branch".to_string());
                }
                match &tokens[p] {
                    Token::Ident(kw) if kw.eq_ignore_ascii_case("else") => {}
                    _ => {
                        return Err(
                            "Inline `if` expression must end with `else <expr>`".to_string()
                        );
                    }
                }
                let (else_expr, p) = parse_add_sub(tokens, p + 1, ctx)?;
                return Ok((
                    Expression::Conditional(
                        Box::new(cond),
                        Box::new(then_expr),
                        Box::new(else_expr),
                    ),
                    p,
                ));
            }

            // Check if it's a function call: name(expr)
            if pos + 1 < tokens.len() && tokens[pos + 1] == Token::LParen {
                let func_name = name.to_lowercase();
                let (arg, p) = parse_add_sub(tokens, pos + 2, ctx)?;
                if p >= tokens.len() || tokens[p] != Token::RParen {
                    return Err(format!("Missing closing parenthesis for function {}", name));
                }
                return Ok((Expression::UnaryFn(func_name, Box::new(arg)), p + 1));
            }

            // Check if it's a theta
            if let Some(idx) = ctx.theta_names.iter().position(|n| n == name) {
                return Ok((Expression::Theta(idx), pos + 1));
            }

            // Check if it's an eta
            if let Some(idx) = ctx.eta_names.iter().position(|n| n == name) {
                return Ok((Expression::Eta(idx), pos + 1));
            }

            // Previously-assigned local variable (e.g. earlier lines of
            // [individual_parameters], or a state/param name injected by the
            // ODE RHS harness).
            if ctx.defined_vars.iter().any(|n| n == name) {
                return Ok((Expression::Variable(name.clone()), pos + 1));
            }

            // Anything else is a covariate reference in the regular model
            // context. The ODE RHS context keeps it as a Variable so that the
            // eval-time `vars` map (which carries state + individual params)
            // can resolve it case-sensitively.
            if ctx.fallback_covariate {
                Ok((Expression::Covariate(name.clone()), pos + 1))
            } else {
                Ok((Expression::Variable(name.clone()), pos + 1))
            }
        }
        other => Err(format!("Unexpected token: {:?}", other)),
    }
}

// ── Condition parser ────────────────────────────────────────────────────────
//
// Precedence (lowest first):  ||  >  &&  >  !  >  comparison.
// The condition grammar lives in its own recursive-descent stack so that the
// arithmetic expression parser doesn't need to understand boolean operators.

pub(crate) fn parse_condition(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Condition, usize), String> {
    parse_cond_or(tokens, pos, ctx)
}

fn parse_cond_or(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Condition, usize), String> {
    let (mut left, mut pos) = parse_cond_and(tokens, pos, ctx)?;
    while pos < tokens.len() && tokens[pos] == Token::OrOr {
        let (right, p) = parse_cond_and(tokens, pos + 1, ctx)?;
        left = Condition::Or(Box::new(left), Box::new(right));
        pos = p;
    }
    Ok((left, pos))
}

fn parse_cond_and(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Condition, usize), String> {
    let (mut left, mut pos) = parse_cond_not(tokens, pos, ctx)?;
    while pos < tokens.len() && tokens[pos] == Token::AndAnd {
        let (right, p) = parse_cond_not(tokens, pos + 1, ctx)?;
        left = Condition::And(Box::new(left), Box::new(right));
        pos = p;
    }
    Ok((left, pos))
}

fn parse_cond_not(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Condition, usize), String> {
    if pos < tokens.len() && tokens[pos] == Token::Bang {
        let (inner, p) = parse_cond_not(tokens, pos + 1, ctx)?;
        return Ok((Condition::Not(Box::new(inner)), p));
    }
    parse_cond_atom(tokens, pos, ctx)
}

fn parse_cond_atom(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Condition, usize), String> {
    // Parenthesised sub-condition: `(cond)`.  Always try to parse the
    // contents as a condition first.  This handles `(a < b)`, `(!(x == 1))`,
    // `((a > 0) && (b < 10))`, etc. without any lookahead heuristic.
    if pos < tokens.len() && tokens[pos] == Token::LParen {
        let (inner, p) = parse_condition(tokens, pos + 1, ctx)?;
        if p >= tokens.len() || tokens[p] != Token::RParen {
            return Err("Missing closing `)` in condition".to_string());
        }
        return Ok((inner, p + 1));
    }

    // comparison: expr <cmpop> expr
    let (lhs, p) = parse_add_sub(tokens, pos, ctx)?;
    if p >= tokens.len() {
        return Err("Expected comparison operator in condition".to_string());
    }
    let op = match tokens[p] {
        Token::Lt => CmpOp::Lt,
        Token::Le => CmpOp::Le,
        Token::Gt => CmpOp::Gt,
        Token::Ge => CmpOp::Ge,
        Token::EqEq => CmpOp::Eq,
        Token::Ne => CmpOp::Ne,
        ref other => {
            return Err(format!(
                "Expected comparison operator (<, <=, >, >=, ==, !=) in condition, got {:?}",
                other
            ));
        }
    };
    let (rhs, p) = parse_add_sub(tokens, p + 1, ctx)?;
    Ok((Condition::Compare(lhs, op, rhs), p))
}
