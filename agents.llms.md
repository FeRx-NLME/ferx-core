# Using ferx from an agent

This page is for a coding agent (or a script) that **uses** ferx: it writes `.ferx` model files, fits them from the CLI or from R, and decides what to do next from the result. It collects the machine-readable surfaces in one place and gives the loop that uses them. (An agent working **on** the ferx-core source reads `AGENTS.md` in the repository instead.)

The site also publishes [`llms.txt`](https://ferx-nlme.org/ferx-core/llms.txt), an index of these pages for language models. Every page has a Markdown twin at the same path with `.html` replaced by `.llms.md`, and `llms-full.txt` concatenates them all.

Three rules cover most of what goes wrong:

- **Do not assume NONMEM syntax.** A `.ferx` file is its own DSL, not a control stream. Read the [model file reference](model-file/index.llms.md) and start from the [quick start](quick-start.llms.md) model rather than translating `$PK` blocks from memory.
- **Branch on codes, never on prose.** Every finding, warning and refusal ferx reports carries a stable token. Messages are for people and are reworded freely.
- **Validate before you fit.** A check costs milliseconds; a fit that fails on a missing covariate column costs a fit.

## The loop

1.  Write or edit the model file, one section at a time ([Edit by section](#edit-by-section)).
2.  Validate it against the data ([Validate before fitting](#validate-before-fitting)). Fix every `error` and go back to step 1.
3.  Optionally run a short pilot fit to see whether the optimizer moves from the initial estimates (`ferx_check_init()` in R).
4.  Fit, asking for the JSON output ([Read the fit as JSON](#read-the-fit-as-json)).
5.  Judge the result from its status and structured warnings ([Judge the fit](#judge-the-fit)), and, in an automated search, gate it ([Gate candidates](#gate-candidates-in-automated-selection)).
6.  Save the run as a [`.fitrx` bundle](file-formats/fitrx.llms.md) so it can be compared later with `ferx summary`.

## Validate before fitting

From the shell:

``` bash
ferx check model.ferx --data data.csv --json
```

The [check report](file-formats/check-report.llms.md) is a JSON object with `valid` and a `diagnostics` array. Each diagnostic has a `severity` (`error`, `warning` or `info`), a stable `code` (`E_*` for errors, `W_*` for warnings), a `message`, and, when ferx can attribute it, the `block`, `line` and a `suggestion`. Branch on:

- the **exit status**: `0` no errors (warnings and info rows alone still exit `0`), `1` at least one error, `2` a usage error such as a missing path;
- the **`code`** of each diagnostic, looked up in the [code tables](file-formats/check-report.llms.md#error-warning-codes). Treat a code you do not recognise as a generic finding of its `severity`;
- **`suggestion`**, when present, as the first fix to try.

Pass `--data` whenever you have the data. Without it only the model-only checks run, and the data-dependent ones (covariate columns present, compartment coverage, steady-state dosing) are what catch most authoring mistakes.

From R, `ferx_model_validate(path, data = NULL)` runs the same parser and checks. It prints a report and invisibly returns a list with `ok` and a `diagnostics` data frame (`severity`, `code`, `message`, `block`, `line`, `suggestion`) carrying the same codes.

## Edit by section

A `.ferx` file is a sequence of `[section]` blocks. Edit it by replacing a whole section, not by applying a regular expression to the file: a regex that matches `CL` also matches `TVCL` and `ETA_CL`.

In R:

- `ferx_model_inspect(path)` prints the parsed structure: parameters, random effects, structural model and options.
- `ferx_model_get_section(path, "error_model")` returns one section’s body.
- `ferx_model_set_section(path, "error_model", lines)` replaces it and leaves every other section untouched. Pass the body lines only, without the `[error_model]` header. A plain path is edited in place; a `ferx_model` pointing at a bundled example is copied to `tempdir()` first.

Validate again after every edit. The section names and their syntax are listed in the [model file reference](model-file/index.llms.md); the `[fit_options]` keys, including `method`, are in [fit options](model-file/fit-options.llms.md).

## Read the fit as JSON

Ask the CLI for the JSON fit file:

``` bash
ferx model.ferx --data data.csv --output-format json --output run1.fitrx
```

`{model}-fit.json` is the complete fit under a [versioned schema](output.llms.md#fit-json-modelfitjson). Prefer it to the YAML, which is a curated view for people. Three points matter to a parser:

- **`schema_version`** is an integer bumped only on a breaking change. Check it before reading anything else; added fields do not bump it.
- **Non-finite values are `null`.** A `NaN` standard error or an infinite condition number arrives as JSON `null`, so test for it rather than assuming every number is present.
- **Read estimates by name** from the top-level [`estimates`](output.llms.md#fit-json-estimates) block, not by position in the `theta` vector.

Library callers get the same payload in-process from `FitResult::to_json_value()`. In R, the `ferx_fit` object carries the same facts as list fields; `ferx_save_fit()` and `ferx_load_fit()` write and read the `.fitrx` bundle.

## Judge the fit

`converged` alone says only that the optimizer’s stop was verified. Read two things instead.

**The status.** [`FitResult::status()`](output.llms.md#fit-status) gives one verdict, `CONVERGED`, `CONVERGED WITH ISSUES` or `NOT CONVERGED`, with a list of reasons such as `evaluation_only`, `budget_spent`, `covariance_failed` or a critical warning code. The JSON carries it as `status`, the YAML as `status:` and `status_reasons:`.

**The structured warnings.** `warnings_structured` is the machine channel; the flat `warnings` strings are the display channel. Each entry has a `severity` (`Critical`, `Warning`, `Info`) and a `category`, a snake_case [`WarningCode`](warnings.llms.md#warning-codes) token that is never renamed or reused once released. The [warnings page](warnings.llms.md#severity) gives a *Typical agent response* for each severity and each code. As a rule, do not accept a model with a `Critical` entry, inspect a `Warning`, and log an `Info`.

In R, `ferx_get_warnings(fit, as_df = TRUE)` returns the same entries as a data frame (`severity`, `category`, `message`, `source_method`), including for a fit loaded back from a `.fitrx` file. `ferx_runlog(fit)` prints a run log that is useful to quote back to a person.

When an entry point refuses to run, the error carries a code too ([matching on the code](warnings.llms.md#entry-point-error-codes)).

## Gate candidates in automated selection

When an agent compares many candidate models, rank only the ones that are fit to rank. A candidate that never left its initial estimates, or has a parameter pinned to a bound, reports an OFV that says nothing about the model.

In R, `check_strictness(fit)` applies those gates and returns `passed`, a `failures` vector naming each gate that failed, and a `skipped` vector for the gates that had nothing to judge (no covariance matrix, for example). The defaults require convergence, reject a theta on a bound and a fit that never moved, and cap the condition number at 1000 and the largest correlation at 0.95. Turn `reject_init_stall` off when candidates are warm-started from a parent’s estimates. Record the `failures` in the search report so a person can see why each candidate was dropped.

The model-search tools apply their own selection rules; see [Tools](tools/index.llms.md).

## Long runs

- **Checkpoints.** The CLI saves a resume point to `{model}.tmp` every `checkpoint_interval_secs` (300 by default). Re-running the same model and data resumes from it; `--clean` starts fresh. See `checkpoint` in [fit options](model-file/fit-options.llms.md#run-control).
- **Background fits in R.** `ferx_fit_async()` starts the fit in a background process and returns a handle; `ferx_collect(handle)` waits for the result and `ferx_stop(handle)` cancels it. In a non-interactive session (`Rscript`, an agent’s subprocess) it runs the fit synchronously instead, so an agent that wants concurrency should start its own processes.
- **Threads.** Set `threads` in `[fit_options]` (or `--threads`) and keep the total across concurrent fits within the machine’s cores.

## Where to look next

| Need | Page |
|----|----|
| Model-file syntax | [Model file reference](model-file/index.llms.md) |
| `[fit_options]` keys | [Fit options](model-file/fit-options.llms.md) |
| Data columns | [Data format](data-format.llms.md) |
| Commands and exit codes | [CLI reference](cli.llms.md) |
| Every check code | [Check report](file-formats/check-report.llms.md) |
| Every warning code | [Warnings](warnings.llms.md) |
| Output files and JSON schema | [Output files](output.llms.md) |
| Choosing an estimation method | [Estimation methods](estimation/index.llms.md) |
| R function reference | [ferx R package reference](https://ferx-nlme.org/reference/) |
