//! The Model Feature Language (MFL): the grammar a `.ferxsearch` file states
//! its search space in (#1179).
//!
//! The grammar is Pharmpy's, reused verbatim so a search space written for
//! Pharmpy's `modelsearch` / `covsearch` / `iivsearch` transfers without
//! translation and so a space is quotable in an issue or a paper:
//!
//! ```text
//! ABSORPTION([FO,ZO]); PERIPHERALS(0..1); LAGTIME([OFF,ON])
//! COVARIATE?(@IIV, @CONTINUOUS, [pow,lin]); COVARIATE?(@IIV, @CATEGORICAL, cat)
//! IIV(@PK, EXP); COVARIANCE(IIV, @PK_IIV)
//! LET(CONTINUOUS, [WT, AGE])
//! ```
//!
//! Statements are separated by `;` or a newline. Keywords, modes and effect
//! names are case-insensitive. A feature takes a comma-separated argument list
//! whose entries are a bare name (`FO`), a number (`1`), a range (`0..2`, both
//! ends included), an array (`[FO, ZO]`), the wildcard `*`, or a symbol
//! (`@IIV`). `LET(NAME, [...])` defines a symbol referenced as `@NAME`; a
//! `?` after the feature keyword marks it exploratory.
//!
//! One deliberate deviation from Pharmpy: parameter names, covariate names and
//! `LET` values keep their case. ferx's `[individual_parameters]` and CSV
//! headers are case-sensitive, and upper-casing `Vc` into `VC` would resolve
//! to nothing.
//!
//! This module only parses and renders. It does not know what ferx can build —
//! that is [`super::coverage`] — and it does not know what `@IIV` means for a
//! given model — that is [`mod@super::resolve`]. Keeping the three apart is what
//! lets an unsupported feature be reported *by name*, after a full parse,
//! instead of as a syntax error.

use std::fmt;

/// A parsed MFL program: its statements in source order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mfl {
    pub statements: Vec<Statement>,
}

/// One `;`- or newline-separated statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// `LET(NAME, [a, b])` — defines `@NAME`.
    Let {
        name: String,
        values: Vec<String>,
    },
    Feature(Feature),
}

/// The feature categories of the grammar.
///
/// Every category Pharmpy's grammar and documentation list is parsed, whether
/// or not ferx can express it, so that `ELIMINATION(MM)` is refused as "not
/// expressible" rather than as "unknown keyword".
#[derive(Debug, Clone, PartialEq)]
pub enum Feature {
    Absorption(Modes<AbsorptionMode>),
    Elimination(Modes<EliminationMode>),
    Peripherals {
        counts: Counts,
        /// `DRUG` / `MET`; `None` when omitted (Pharmpy's default is `DRUG`).
        kind: Option<Modes<PeripheralKind>>,
    },
    Transits {
        counts: TransitCounts,
        /// `DEPOT` / `NODEPOT`; `None` when omitted.
        depot: Option<Modes<DepotMode>>,
    },
    Lagtime(Modes<LagtimeMode>),
    Covariate {
        optional: bool,
        parameters: Operand,
        covariates: Operand,
        effects: Modes<CovariateEffect>,
        op: CovariateOp,
    },
    Allometry {
        covariate: String,
        /// The reference value (`ALLOMETRY(WT, 70)`); `None` when omitted.
        reference: Option<String>,
    },
    DirectEffect(Modes<PdMode>),
    EffectComp(Modes<PdMode>),
    IndirectEffect {
        mode: Modes<PdMode>,
        production: Modes<ProductionMode>,
    },
    Metabolite(Modes<MetaboliteMode>),
    Iiv {
        optional: bool,
        parameters: Operand,
        effects: Modes<VariabilityEffect>,
    },
    Iov {
        optional: bool,
        parameters: Operand,
        effects: Modes<VariabilityEffect>,
    },
    Covariance {
        optional: bool,
        level: VariabilityLevel,
        parameters: Operand,
    },
}

impl Feature {
    /// The keyword this feature is written with, `?` included when exploratory.
    pub fn keyword(&self) -> String {
        let (word, optional) = match self {
            Feature::Absorption(_) => ("ABSORPTION", false),
            Feature::Elimination(_) => ("ELIMINATION", false),
            Feature::Peripherals { .. } => ("PERIPHERALS", false),
            Feature::Transits { .. } => ("TRANSITS", false),
            Feature::Lagtime(_) => ("LAGTIME", false),
            Feature::Covariate { optional, .. } => ("COVARIATE", *optional),
            Feature::Allometry { .. } => ("ALLOMETRY", false),
            Feature::DirectEffect(_) => ("DIRECTEFFECT", false),
            Feature::EffectComp(_) => ("EFFECTCOMP", false),
            Feature::IndirectEffect { .. } => ("INDIRECTEFFECT", false),
            Feature::Metabolite(_) => ("METABOLITE", false),
            Feature::Iiv { optional, .. } => ("IIV", *optional),
            Feature::Iov { optional, .. } => ("IOV", *optional),
            Feature::Covariance { optional, .. } => ("COVARIANCE", *optional),
        };
        if optional {
            format!("{word}?")
        } else {
            word.to_string()
        }
    }
}

/// A mode argument: an explicit set, or `*` for every mode the grammar knows.
#[derive(Debug, Clone, PartialEq)]
pub enum Modes<M> {
    Wildcard,
    List(Vec<M>),
}

impl<M: Mode> Modes<M> {
    /// The modes this stands for, with `*` expanded to [`Mode::ALL`].
    pub fn expand(&self) -> Vec<M> {
        match self {
            Modes::Wildcard => M::ALL.to_vec(),
            Modes::List(list) => list.clone(),
        }
    }
}

/// A closed set of keyword values, spelled case-insensitively.
pub trait Mode: Copy + PartialEq + fmt::Debug + 'static {
    /// Every value, in the order `*` expands to.
    const ALL: &'static [Self];
    /// The canonical (rendered) spelling.
    fn label(&self) -> &'static str;
    /// Parse one spelling, case-insensitively.
    fn parse(word: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|m| m.label().eq_ignore_ascii_case(word))
    }
    /// What the grammar calls this set, for error messages.
    const WHAT: &'static str;
}

macro_rules! mode_enum {
    ($(#[$meta:meta])* $name:ident, $what:literal, { $($variant:ident => $label:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name {
            $($variant),+
        }

        impl Mode for $name {
            const ALL: &'static [$name] = &[$($name::$variant),+];
            const WHAT: &'static str = $what;
            fn label(&self) -> &'static str {
                match self {
                    $($name::$variant => $label),+
                }
            }
        }
    };
}

mode_enum!(
    /// `ABSORPTION(...)`.
    AbsorptionMode, "absorption mode", {
        Inst => "INST", Fo => "FO", Zo => "ZO", SeqZoFo => "SEQ-ZO-FO", Weibull => "WEIBULL",
    }
);
mode_enum!(
    /// `ELIMINATION(...)`.
    EliminationMode, "elimination mode", {
        Fo => "FO", Zo => "ZO", Mm => "MM", MixFoMm => "MIX-FO-MM",
    }
);
mode_enum!(
    /// The second argument of `PERIPHERALS(n, ...)`.
    PeripheralKind, "peripheral kind", { Drug => "DRUG", Met => "MET" }
);
mode_enum!(
    /// The second argument of `TRANSITS(n, ...)`.
    DepotMode, "depot option", { Depot => "DEPOT", NoDepot => "NODEPOT" }
);
mode_enum!(
    /// `LAGTIME(...)`.
    LagtimeMode, "lagtime option", { Off => "OFF", On => "ON" }
);
mode_enum!(
    /// A covariate effect function. The wildcard expands to the four
    /// continuous forms, as in Pharmpy; `cat`, `cat2` and `custom` must be
    /// named.
    CovariateEffect, "covariate effect", {
        Lin => "lin", PieceLin => "piece_lin", Exp => "exp", Pow => "pow",
        Cat => "cat", Cat2 => "cat2", Custom => "custom",
    }
);
mode_enum!(
    /// `DIRECTEFFECT` / `EFFECTCOMP` / `INDIRECTEFFECT` model types.
    PdMode, "PD effect type", {
        Linear => "LINEAR", Emax => "EMAX", Sigmoid => "SIGMOID", Step => "STEP", Loglin => "LOGLIN",
    }
);
mode_enum!(
    /// The second argument of `INDIRECTEFFECT(type, ...)`.
    ProductionMode, "indirect-effect kind", { Production => "PRODUCTION", Degradation => "DEGRADATION" }
);
mode_enum!(
    /// `METABOLITE(...)`.
    MetaboliteMode, "metabolite model", { Basic => "BASIC", Psc => "PSC" }
);
mode_enum!(
    /// How an η enters a parameter in `IIV(p, effect)` / `IOV(p, effect)`.
    VariabilityEffect, "variability effect", {
        Exp => "EXP", Add => "ADD", Prop => "PROP", Log => "LOG", ReLog => "RE_LOG",
    }
);
mode_enum!(
    /// The first argument of `COVARIANCE(level, params)`.
    VariabilityLevel, "variability level", { Iiv => "IIV", Iov => "IOV" }
);

impl CovariateEffect {
    /// Pharmpy expands `*` to the continuous forms only.
    pub const CONTINUOUS: &'static [CovariateEffect] = &[
        CovariateEffect::Lin,
        CovariateEffect::PieceLin,
        CovariateEffect::Exp,
        CovariateEffect::Pow,
    ];

    /// Whether this effect reads a categorical covariate.
    pub fn is_categorical(&self) -> bool {
        matches!(self, CovariateEffect::Cat | CovariateEffect::Cat2)
    }
}

/// The operator a covariate effect is combined with: `*` (default) or `+`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CovariateOp {
    Multiply,
    Add,
}

impl CovariateOp {
    pub fn label(&self) -> &'static str {
        match self {
            CovariateOp::Multiply => "*",
            CovariateOp::Add => "+",
        }
    }
}

/// A parameter or covariate argument: explicit names, a `@`-symbol, or `*`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operand {
    Wildcard,
    Symbol(String),
    Names(Vec<String>),
}

impl Operand {
    /// `true` when the operand needs a model to say what it stands for.
    pub fn is_symbolic(&self) -> bool {
        !matches!(self, Operand::Names(_))
    }
}

/// A count argument (`PERIPHERALS`, `TRANSITS`): one number, a range, or an
/// array of numbers. The source shape is kept so rendering is faithful.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Counts {
    Single(u32),
    Range(u32, u32),
    List(Vec<u32>),
}

impl Counts {
    /// Every count this stands for, ascending and de-duplicated.
    pub fn expand(&self) -> Vec<u32> {
        let mut v: Vec<u32> = match self {
            Counts::Single(n) => vec![*n],
            Counts::Range(a, b) => (*a..=*b).collect(),
            Counts::List(list) => list.clone(),
        };
        v.sort_unstable();
        v.dedup();
        v
    }
}

/// `TRANSITS` takes either counts or the literal `N` (estimate the number of
/// transit compartments as a continuous parameter).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TransitCounts {
    N,
    Counts(Counts),
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

impl Mfl {
    /// Render back to MFL text, one statement per `;`-separated entry.
    ///
    /// The rendering re-parses to an equal [`Mfl`] (the round-trip the
    /// `mfl_tests` pin), which is what makes a space a stable key: the same
    /// features always render to the same string, whatever the source
    /// spacing or casing was.
    pub fn render(&self) -> String {
        self.statements
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(";")
    }

    /// The feature statements, `LET` definitions skipped.
    pub fn features(&self) -> impl Iterator<Item = &Feature> {
        self.statements.iter().filter_map(|s| match s {
            Statement::Feature(f) => Some(f),
            Statement::Let { .. } => None,
        })
    }
}

impl fmt::Display for Mfl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Let { name, values } => {
                write!(f, "LET({name},{})", render_names(values))
            }
            Statement::Feature(feature) => feature.fmt(f),
        }
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.keyword())?;
        match self {
            Feature::Absorption(m) => write!(f, "{}", render_modes(m))?,
            Feature::Elimination(m) => write!(f, "{}", render_modes(m))?,
            Feature::Peripherals { counts, kind } => {
                write!(f, "{counts}")?;
                if let Some(kind) = kind {
                    write!(f, ",{}", render_modes(kind))?;
                }
            }
            Feature::Transits { counts, depot } => {
                match counts {
                    TransitCounts::N => write!(f, "N")?,
                    TransitCounts::Counts(c) => write!(f, "{c}")?,
                }
                if let Some(depot) = depot {
                    write!(f, ",{}", render_modes(depot))?;
                }
            }
            Feature::Lagtime(m) => write!(f, "{}", render_modes(m))?,
            Feature::Covariate {
                parameters,
                covariates,
                effects,
                op,
                ..
            } => {
                write!(f, "{parameters},{covariates},{}", render_modes(effects))?;
                if *op == CovariateOp::Add {
                    write!(f, ",+")?;
                }
            }
            Feature::Allometry {
                covariate,
                reference,
            } => {
                write!(f, "{covariate}")?;
                if let Some(r) = reference {
                    write!(f, ",{r}")?;
                }
            }
            Feature::DirectEffect(m) => write!(f, "{}", render_modes(m))?,
            Feature::EffectComp(m) => write!(f, "{}", render_modes(m))?,
            Feature::IndirectEffect { mode, production } => {
                write!(f, "{},{}", render_modes(mode), render_modes(production))?
            }
            Feature::Metabolite(m) => write!(f, "{}", render_modes(m))?,
            Feature::Iiv {
                parameters,
                effects,
                ..
            }
            | Feature::Iov {
                parameters,
                effects,
                ..
            } => write!(f, "{parameters},{}", render_modes(effects))?,
            Feature::Covariance {
                level, parameters, ..
            } => write!(f, "{},{parameters}", level.label())?,
        }
        write!(f, ")")
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Wildcard => f.write_str("*"),
            Operand::Symbol(name) => write!(f, "@{name}"),
            Operand::Names(names) => f.write_str(&render_names(names)),
        }
    }
}

impl fmt::Display for Counts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Counts::Single(n) => write!(f, "{n}"),
            Counts::Range(a, b) => write!(f, "{a}..{b}"),
            Counts::List(list) => {
                let items: Vec<String> = list.iter().map(|n| n.to_string()).collect();
                write!(f, "[{}]", items.join(","))
            }
        }
    }
}

fn render_modes<M: Mode>(modes: &Modes<M>) -> String {
    match modes {
        Modes::Wildcard => "*".to_string(),
        Modes::List(list) if list.len() == 1 => list[0].label().to_string(),
        Modes::List(list) => {
            let items: Vec<&str> = list.iter().map(|m| m.label()).collect();
            format!("[{}]", items.join(","))
        }
    }
}

fn render_names(names: &[String]) -> String {
    if names.len() == 1 {
        names[0].clone()
    } else {
        format!("[{}]", names.join(","))
    }
}

// ---------------------------------------------------------------------------
// Lexing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// A bare word: letters, digits, `_`, `-` (for `SEQ-ZO-FO`), starting
    /// with a letter or `_`.
    Word(String),
    /// A number literal, kept as written so `70.5` survives rendering.
    Number(String),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    DotDot,
    Star,
    Plus,
    Question,
    At,
    /// `;` or a newline.
    Separator,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Word(w) => write!(f, "`{w}`"),
            Token::Number(n) => write!(f, "`{n}`"),
            Token::LParen => f.write_str("`(`"),
            Token::RParen => f.write_str("`)`"),
            Token::LBracket => f.write_str("`[`"),
            Token::RBracket => f.write_str("`]`"),
            Token::Comma => f.write_str("`,`"),
            Token::DotDot => f.write_str("`..`"),
            Token::Star => f.write_str("`*`"),
            Token::Plus => f.write_str("`+`"),
            Token::Question => f.write_str("`?`"),
            Token::At => f.write_str("`@`"),
            Token::Separator => f.write_str("end of statement"),
        }
    }
}

fn lex(src: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = src.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\r' => i += 1,
            '\n' | ';' => {
                tokens.push(Token::Separator);
                i += 1;
            }
            '#' => {
                // A comment runs to the end of the line; the newline itself is
                // still a separator so `A(x)  # note\nB(y)` is two statements.
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
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
            '[' => {
                tokens.push(Token::LBracket);
                i += 1;
            }
            ']' => {
                tokens.push(Token::RBracket);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '?' => {
                tokens.push(Token::Question);
                i += 1;
            }
            '@' => {
                tokens.push(Token::At);
                i += 1;
            }
            '.' if chars.get(i + 1) == Some(&'.') => {
                tokens.push(Token::DotDot);
                i += 2;
            }
            c if c.is_ascii_digit() => {
                let start = i;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                // A decimal point followed by a digit belongs to the number;
                // `..` (a range) does not.
                if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
                    i += 1;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                tokens.push(Token::Number(chars[start..i].iter().collect()));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_alphanumeric() || chars[i] == '_' || chars[i] == '-')
                {
                    i += 1;
                }
                tokens.push(Token::Word(chars[start..i].iter().collect()));
            }
            other => {
                return Err(format!(
                    "MFL: unexpected character `{other}` at offset {i}: \
                     expected a feature keyword, `(`, `)`, `[`, `]`, `,`, `*`, `?`, `@`, \
                     `..`, `;` or a newline"
                ));
            }
        }
    }
    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// One raw argument of a feature call, before the feature says what it means.
#[derive(Debug, Clone, PartialEq)]
enum Arg {
    Wildcard,
    Symbol(String),
    Word(String),
    Number(String),
    Range(u32, u32),
    List(Vec<Item>),
    Plus,
}

impl Arg {
    fn describe(&self) -> String {
        match self {
            Arg::Wildcard => "`*`".into(),
            Arg::Symbol(s) => format!("`@{s}`"),
            Arg::Word(w) => format!("`{w}`"),
            Arg::Number(n) => format!("`{n}`"),
            Arg::Range(a, b) => format!("`{a}..{b}`"),
            Arg::List(_) => "an array".into(),
            Arg::Plus => "`+`".into(),
        }
    }
}

/// An entry of an array argument.
#[derive(Debug, Clone, PartialEq)]
enum Item {
    Word(String),
    Number(String),
}

impl Item {
    fn text(&self) -> &str {
        match self {
            Item::Word(w) | Item::Number(w) => w,
        }
    }
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    fn expect(&mut self, want: Token, context: &str) -> Result<(), String> {
        match self.next() {
            Some(t) if t == want => Ok(()),
            Some(t) => Err(format!("MFL: in {context}: expected {want}, found {t}")),
            None => Err(format!(
                "MFL: in {context}: expected {want}, found end of input"
            )),
        }
    }

    fn skip_separators(&mut self) {
        while self.peek() == Some(&Token::Separator) {
            self.pos += 1;
        }
    }

    fn statement(&mut self) -> Result<Statement, String> {
        let keyword = match self.next() {
            Some(Token::Word(w)) => w,
            Some(t) => return Err(format!("MFL: expected a feature keyword, found {t}")),
            None => return Err("MFL: expected a feature keyword, found end of input".into()),
        };
        let upper = keyword.to_ascii_uppercase();
        if upper == "LET" {
            return self.let_statement();
        }
        let optional = if self.peek() == Some(&Token::Question) {
            self.pos += 1;
            true
        } else {
            false
        };
        let context = if optional {
            format!("{upper}?")
        } else {
            upper.clone()
        };
        self.expect(Token::LParen, &context)?;
        let args = self.args(&context)?;
        self.expect(Token::RParen, &context)?;
        let feature = build_feature(&upper, optional, args, &keyword)?;
        Ok(Statement::Feature(feature))
    }

    fn let_statement(&mut self) -> Result<Statement, String> {
        self.expect(Token::LParen, "LET")?;
        let name = match self.next() {
            Some(Token::Word(w)) => w,
            Some(t) => return Err(format!("MFL: in LET: expected a symbol name, found {t}")),
            None => return Err("MFL: in LET: expected a symbol name, found end of input".into()),
        };
        self.expect(Token::Comma, "LET")?;
        let arg = self.arg("LET")?;
        self.expect(Token::RParen, "LET")?;
        let values = names_of(&arg, "LET", "a value or an array of values")?;
        Ok(Statement::Let { name, values })
    }

    fn args(&mut self, context: &str) -> Result<Vec<Arg>, String> {
        let mut args = Vec::new();
        if self.peek() == Some(&Token::RParen) {
            return Ok(args);
        }
        loop {
            args.push(self.arg(context)?);
            if self.peek() == Some(&Token::Comma) {
                self.pos += 1;
            } else {
                return Ok(args);
            }
        }
    }

    fn arg(&mut self, context: &str) -> Result<Arg, String> {
        match self.next() {
            Some(Token::Star) => Ok(Arg::Wildcard),
            Some(Token::Plus) => Ok(Arg::Plus),
            Some(Token::At) => match self.next() {
                Some(Token::Word(w)) => Ok(Arg::Symbol(w)),
                Some(t) => Err(format!(
                    "MFL: in {context}: expected a symbol name after `@`, found {t}"
                )),
                None => Err(format!(
                    "MFL: in {context}: expected a symbol name after `@`, found end of input"
                )),
            },
            Some(Token::Word(w)) => Ok(Arg::Word(w)),
            Some(Token::Number(n)) => {
                if self.peek() == Some(&Token::DotDot) {
                    self.pos += 1;
                    let hi = match self.next() {
                        Some(Token::Number(h)) => h,
                        Some(t) => {
                            return Err(format!(
                                "MFL: in {context}: expected a number after `..`, found {t}"
                            ))
                        }
                        None => {
                            return Err(format!(
                            "MFL: in {context}: expected a number after `..`, found end of input"
                        ))
                        }
                    };
                    let lo = count_of(&n, context)?;
                    let hi = count_of(&hi, context)?;
                    if hi < lo {
                        return Err(format!(
                            "MFL: in {context}: range `{lo}..{hi}` runs backwards"
                        ));
                    }
                    Ok(Arg::Range(lo, hi))
                } else {
                    Ok(Arg::Number(n))
                }
            }
            Some(Token::LBracket) => {
                let mut items = Vec::new();
                if self.peek() == Some(&Token::RBracket) {
                    self.pos += 1;
                    return Ok(Arg::List(items));
                }
                loop {
                    match self.next() {
                        Some(Token::Word(w)) => items.push(Item::Word(w)),
                        Some(Token::Number(n)) => items.push(Item::Number(n)),
                        Some(t) => {
                            return Err(format!(
                                "MFL: in {context}: expected a name or number inside `[...]`, found {t}"
                            ))
                        }
                        None => {
                            return Err(format!(
                                "MFL: in {context}: unclosed `[` at end of input"
                            ))
                        }
                    }
                    match self.next() {
                        Some(Token::Comma) => continue,
                        Some(Token::RBracket) => return Ok(Arg::List(items)),
                        Some(t) => {
                            return Err(format!(
                                "MFL: in {context}: expected `,` or `]` inside an array, found {t}"
                            ))
                        }
                        None => {
                            return Err(format!("MFL: in {context}: unclosed `[` at end of input"))
                        }
                    }
                }
            }
            Some(t) => Err(format!(
                "MFL: in {context}: unexpected {t} in argument list"
            )),
            None => Err(format!("MFL: in {context}: unclosed `(` at end of input")),
        }
    }
}

fn count_of(text: &str, context: &str) -> Result<u32, String> {
    text.parse::<u32>()
        .map_err(|_| format!("MFL: in {context}: `{text}` is not a non-negative integer count"))
}

/// A word, or an array of words, as a list of names (case preserved).
fn names_of(arg: &Arg, context: &str, want: &str) -> Result<Vec<String>, String> {
    match arg {
        Arg::Word(w) => Ok(vec![w.clone()]),
        Arg::Number(n) => Ok(vec![n.clone()]),
        Arg::List(items) => {
            if items.is_empty() {
                return Err(format!(
                    "MFL: in {context}: an empty array `[]` names nothing"
                ));
            }
            Ok(items.iter().map(|i| i.text().to_string()).collect())
        }
        other => Err(format!(
            "MFL: in {context}: expected {want}, found {}",
            other.describe()
        )),
    }
}

fn modes_of<M: Mode>(arg: &Arg, context: &str) -> Result<Modes<M>, String> {
    let known = || {
        M::ALL
            .iter()
            .map(|m| m.label())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let one = |word: &str| -> Result<M, String> {
        M::parse(word).ok_or_else(|| {
            format!(
                "MFL: in {context}: `{word}` is not a known {}; known: {}",
                M::WHAT,
                known()
            )
        })
    };
    match arg {
        Arg::Wildcard => Ok(Modes::Wildcard),
        Arg::Word(w) => Ok(Modes::List(vec![one(w)?])),
        Arg::List(items) => {
            if items.is_empty() {
                return Err(format!(
                    "MFL: in {context}: an empty array `[]` names no {}",
                    M::WHAT
                ));
            }
            let mut list = Vec::with_capacity(items.len());
            for item in items {
                let m = one(item.text())?;
                if !list.contains(&m) {
                    list.push(m);
                }
            }
            Ok(Modes::List(list))
        }
        other => Err(format!(
            "MFL: in {context}: expected a {}, an array of them, or `*`; found {}",
            M::WHAT,
            other.describe()
        )),
    }
}

fn operand_of(arg: &Arg, context: &str, what: &str) -> Result<Operand, String> {
    match arg {
        Arg::Wildcard => Ok(Operand::Wildcard),
        Arg::Symbol(s) => Ok(Operand::Symbol(s.clone())),
        Arg::Word(_) | Arg::Number(_) | Arg::List(_) => Ok(Operand::Names(names_of(
            arg,
            context,
            &format!("a {what} name, an array of them, `*` or `@SYMBOL`"),
        )?)),
        other => Err(format!(
            "MFL: in {context}: expected a {what} name, an array of them, `*` or `@SYMBOL`; found {}",
            other.describe()
        )),
    }
}

fn counts_of(arg: &Arg, context: &str) -> Result<Counts, String> {
    match arg {
        Arg::Number(n) => Ok(Counts::Single(count_of(n, context)?)),
        Arg::Range(a, b) => Ok(Counts::Range(*a, *b)),
        Arg::List(items) => {
            if items.is_empty() {
                return Err(format!(
                    "MFL: in {context}: an empty array `[]` gives no count"
                ));
            }
            let mut list = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Item::Number(n) => list.push(count_of(n, context)?),
                    Item::Word(w) => {
                        return Err(format!(
                            "MFL: in {context}: `{w}` is not a count; expected an array of integers"
                        ))
                    }
                }
            }
            Ok(Counts::List(list))
        }
        other => Err(format!(
            "MFL: in {context}: expected a count, a range `a..b`, or an array of counts; found {}",
            other.describe()
        )),
    }
}

fn arity(context: &str, args: &[Arg], min: usize, max: usize, shape: &str) -> Result<(), String> {
    if args.len() < min || args.len() > max {
        Err(format!(
            "MFL: {context} takes {shape}; found {} argument{}",
            args.len(),
            if args.len() == 1 { "" } else { "s" }
        ))
    } else {
        Ok(())
    }
}

fn build_feature(
    upper: &str,
    optional: bool,
    args: Vec<Arg>,
    as_written: &str,
) -> Result<Feature, String> {
    let ctx = if optional {
        format!("{upper}?")
    } else {
        upper.to_string()
    };
    let no_optional = |feature: Feature| -> Result<Feature, String> {
        if optional {
            Err(format!(
                "MFL: `{upper}?` is not valid — only COVARIATE, IIV, IOV and COVARIANCE take the \
                 exploratory `?`"
            ))
        } else {
            Ok(feature)
        }
    };
    match upper {
        "ABSORPTION" => {
            arity(
                &ctx,
                &args,
                1,
                1,
                "one argument (a mode, an array of modes, or `*`)",
            )?;
            no_optional(Feature::Absorption(modes_of(&args[0], &ctx)?))
        }
        "ELIMINATION" => {
            arity(
                &ctx,
                &args,
                1,
                1,
                "one argument (a mode, an array of modes, or `*`)",
            )?;
            no_optional(Feature::Elimination(modes_of(&args[0], &ctx)?))
        }
        "PERIPHERALS" => {
            arity(
                &ctx,
                &args,
                1,
                2,
                "a count and an optional DRUG/MET argument",
            )?;
            let counts = counts_of(&args[0], &ctx)?;
            let kind = match args.get(1) {
                Some(a) => Some(modes_of(a, &ctx)?),
                None => None,
            };
            no_optional(Feature::Peripherals { counts, kind })
        }
        "TRANSITS" => {
            arity(
                &ctx,
                &args,
                1,
                2,
                "a count (or `N`) and an optional DEPOT/NODEPOT argument",
            )?;
            let counts = match &args[0] {
                Arg::Word(w) if w.eq_ignore_ascii_case("N") => {
                    // Pharmpy's grammar: `TRANSITS(N)` takes no depot option.
                    if args.len() > 1 {
                        return Err(format!(
                            "MFL: in {ctx}: `TRANSITS(N)` takes no second argument — \
                             estimating the count implies NODEPOT"
                        ));
                    }
                    TransitCounts::N
                }
                other => TransitCounts::Counts(counts_of(other, &ctx)?),
            };
            let depot = match args.get(1) {
                Some(a) => Some(modes_of(a, &ctx)?),
                None => None,
            };
            no_optional(Feature::Transits { counts, depot })
        }
        "LAGTIME" => {
            arity(
                &ctx,
                &args,
                1,
                1,
                "one argument (ON, OFF, an array of them, or `*`)",
            )?;
            no_optional(Feature::Lagtime(modes_of(&args[0], &ctx)?))
        }
        "COVARIATE" => {
            arity(
                &ctx,
                &args,
                3,
                4,
                "parameter(s), covariate(s), effect(s) and an optional `+`/`*` operator",
            )?;
            let parameters = operand_of(&args[0], &ctx, "parameter")?;
            let covariates = operand_of(&args[1], &ctx, "covariate")?;
            let effects = modes_of(&args[2], &ctx)?;
            // Pharmpy: "Mandatory effects need to be explicit (not '*')". A
            // forced relation is in every candidate, so the search has no
            // way to pick a form for it.
            if !optional && effects == Modes::Wildcard {
                return Err(format!(
                    "MFL: in {ctx}: a mandatory (non-`?`) COVARIATE needs explicit effects, \
                     not `*` — write `COVARIATE?` to search over the forms, or name one"
                ));
            }
            let op = match args.get(3) {
                None => CovariateOp::Multiply,
                Some(Arg::Plus) => CovariateOp::Add,
                Some(Arg::Wildcard) => CovariateOp::Multiply,
                Some(other) => {
                    return Err(format!(
                        "MFL: in {ctx}: the fourth argument is the operator, `+` or `*`; found {}",
                        other.describe()
                    ))
                }
            };
            Ok(Feature::Covariate {
                optional,
                parameters,
                covariates,
                effects,
                op,
            })
        }
        "ALLOMETRY" => {
            arity(
                &ctx,
                &args,
                1,
                2,
                "a covariate name and an optional reference value",
            )?;
            let covariate = match &args[0] {
                Arg::Word(w) => w.clone(),
                other => {
                    return Err(format!(
                        "MFL: in {ctx}: expected a covariate name, found {}",
                        other.describe()
                    ))
                }
            };
            let reference = match args.get(1) {
                None => None,
                Some(Arg::Number(n)) => Some(n.clone()),
                Some(other) => {
                    return Err(format!(
                        "MFL: in {ctx}: the reference value must be a number, found {}",
                        other.describe()
                    ))
                }
            };
            no_optional(Feature::Allometry {
                covariate,
                reference,
            })
        }
        "DIRECTEFFECT" => {
            arity(
                &ctx,
                &args,
                1,
                1,
                "one argument (a type, an array of types, or `*`)",
            )?;
            no_optional(Feature::DirectEffect(modes_of(&args[0], &ctx)?))
        }
        "EFFECTCOMP" => {
            arity(
                &ctx,
                &args,
                1,
                1,
                "one argument (a type, an array of types, or `*`)",
            )?;
            no_optional(Feature::EffectComp(modes_of(&args[0], &ctx)?))
        }
        "INDIRECTEFFECT" => {
            arity(&ctx, &args, 2, 2, "a type and PRODUCTION/DEGRADATION")?;
            no_optional(Feature::IndirectEffect {
                mode: modes_of(&args[0], &ctx)?,
                production: modes_of(&args[1], &ctx)?,
            })
        }
        "METABOLITE" => {
            arity(
                &ctx,
                &args,
                1,
                1,
                "one argument (BASIC, PSC, an array of them, or `*`)",
            )?;
            no_optional(Feature::Metabolite(modes_of(&args[0], &ctx)?))
        }
        "IIV" | "IOV" => {
            arity(&ctx, &args, 2, 2, "parameter(s) and effect(s)")?;
            let parameters = operand_of(&args[0], &ctx, "parameter")?;
            let effects = modes_of(&args[1], &ctx)?;
            Ok(if upper == "IIV" {
                Feature::Iiv {
                    optional,
                    parameters,
                    effects,
                }
            } else {
                Feature::Iov {
                    optional,
                    parameters,
                    effects,
                }
            })
        }
        "COVARIANCE" => {
            arity(&ctx, &args, 2, 2, "IIV or IOV, then parameter(s)")?;
            let level = match &args[0] {
                Arg::Word(w) => VariabilityLevel::parse(w).ok_or_else(|| {
                    format!("MFL: in {ctx}: the first argument must be IIV or IOV, found `{w}`")
                })?,
                other => {
                    return Err(format!(
                        "MFL: in {ctx}: the first argument must be IIV or IOV, found {}",
                        other.describe()
                    ))
                }
            };
            let parameters = operand_of(&args[1], &ctx, "parameter")?;
            Ok(Feature::Covariance {
                optional,
                level,
                parameters,
            })
        }
        _ => Err(format!(
            "MFL: `{as_written}` is not a feature keyword; known: ABSORPTION, ELIMINATION, \
             PERIPHERALS, TRANSITS, LAGTIME, COVARIATE, ALLOMETRY, DIRECTEFFECT, EFFECTCOMP, \
             INDIRECTEFFECT, METABOLITE, IIV, IOV, COVARIANCE, LET"
        )),
    }
}

impl Mfl {
    /// Parse MFL text. An empty (or comment-only) string is an empty space,
    /// which the caller decides about — a search over nothing is its call.
    pub fn parse(src: &str) -> Result<Mfl, String> {
        let tokens = lex(src)?;
        let mut parser = Parser { tokens, pos: 0 };
        let mut statements = Vec::new();
        loop {
            parser.skip_separators();
            if parser.peek().is_none() {
                break;
            }
            statements.push(parser.statement()?);
            match parser.peek() {
                None | Some(Token::Separator) => {}
                Some(t) => {
                    return Err(format!(
                        "MFL: expected `;` or a newline after a statement, found {t}"
                    ))
                }
            }
        }
        Ok(Mfl { statements })
    }
}

#[cfg(test)]
#[path = "mfl_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "mfl_pharmpy_anchor.rs"]
mod pharmpy_anchor;
