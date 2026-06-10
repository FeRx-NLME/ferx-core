# Security Policy

## Supported versions

ferx-core is pre-1.0 and under active development. Security fixes are applied to
the **latest release** (and `main`); older release lines are not maintained.
Please upgrade to the latest tagged release before reporting an issue.

## Reporting a vulnerability

**Please do not open a public GitHub Issue for security vulnerabilities.**

Report privately via GitHub's
[**Report a vulnerability**](https://github.com/FeRx-NLME/ferx-core/security/advisories/new)
flow (Security → Advisories → Report a vulnerability). If you can't use that,
contact a maintainer directly.

Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal `.ferx` model / data subset if relevant),
- the affected version(s) or commit, and
- any suggested fix.

We aim to acknowledge a report within a few business days, agree on a disclosure
timeline, and credit reporters who wish to be named once a fix ships.

## Scope

ferx-core is a numerical library, not a network service, so its main security
surface is:

- **Native / FFI code** — the Rust engine is consumed from R (ferx-r) through
  the extendr FFI boundary; memory-safety bugs there can crash the host process.
- **Dependencies** — Rust crates (monitored by `cargo audit` and Dependabot) and
  the Enzyme toolchain used for autodiff builds.
- **Untrusted input** — parsing `.ferx` model files and NONMEM-format CSV data.

See the [Security & dependencies](https://ferx-nlme.github.io/ferx-core/development/sdlc.html#14-security--dependencies)
section of the development docs for how this fits into the wider process.
