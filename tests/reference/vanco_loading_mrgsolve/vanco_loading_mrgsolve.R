#!/usr/bin/env Rscript
# Vancomycin LOADING-DOSE + maintenance-titration reference (epic #391, issue #702).
#
# External comparator for ferx-core's reactive `[adaptive_dosing]` **pre-scheduled
# base regimen** support (#702): the subject starts on a fixed LOADING DOSE, and the
# controller then titrates DAILY MAINTENANCE doses on the measured trough. NONMEM has
# no native feedback dosing, so mrgsolve (which does) is the apples-to-apples anchor
# for this feature family (see docs/model-file/adaptive-dosing.qmd, "Validation").
#
# The ferx model under test is examples/adaptive_vanco_loading.ferx; the loading dose
# rides on the subject (tests/reference/vanco_loading_mrgsolve/vanco_loading_subject.csv)
# and the [adaptive_dosing] block drives the maintenance decisions. This script
# reproduces the identical structural model + loading dose + controller in mrgsolve and
# freezes the realized maintenance ladder (troughs + doses) to expected.md, which
# tests/adaptive_vanco_loading_anchor.rs asserts ferx matches.
#
# Scenario: a 1-compartment IV vancomycin model. A 1500 mg loading BOLUS at t=0 (the
# pre-scheduled base regimen) seeds the compartment; the controller then reads the
# pre-dose trough (CENT/V) at the start of each subsequent day and titrates the
# maintenance bolus +/-25% to keep the trough in [10, 15] mg/L. The loading dose is
# what makes the first maintenance decision non-trivial (the trough at day 1 is the
# decayed loading dose, ~5.6 mg/L, not 0) -- so a dropped or mis-integrated base
# regimen would move every trough and the anchor would fail.
#
# The R loop mirrors ferx's controller semantics EXACTLY (confirm = 1):
#   - read the latent signal (CENT/V) at the decision time (pre-dose trough),
#   - first-matching rule wins (increase listed before decrease),
#   - `increase 25%` / `decrease 25%` scale the running dose by 1.25 / 0.75 and clamp
#     to dose_bounds, no rule match -> re-issue the running dose unchanged.
#
# Regenerate:  Rscript vanco_loading_mrgsolve.R   (run from this directory)
# Requires:    mrgsolve (tested with 1.7.2) + a C/C++ toolchain (JIT compile).

suppressMessages(library(mrgsolve))

# ---- structural model: identical params to adaptive_vanco_loading.ferx -------
code <- '
$PARAM CL=4.0, V=80.0
$CMT CENT
$ODE
  dxdt_CENT = -(CL/V)*CENT;
'
mod <- mcode("vanco_loading", code, rtol = 1e-10, atol = 1e-10, maxsteps = 100000)

V          <- 80.0                # central volume (L); CENT/V is the concentration
cycle_h    <- 24                  # once-daily maintenance interval
load_dose  <- 1500                # pre-scheduled LOADING bolus at t=0 (#702 base regimen)
n_dec      <- 6                   # 6 daily maintenance decisions, t = 24 .. 144 h
dec_t      <- seq(cycle_h, by = cycle_h, length.out = n_dec)  # 24, 48, ..., 144
start_dose <- 750                 # empiric maintenance start (mg/day)
lo_bound   <- 250                 # dose_bounds
hi_bound   <- 4000
trough_lo  <- 10                  # trough < 10 -> increase 25%
trough_hi  <- 15                  # trough > 15 -> decrease 25%

# ---- loading dose: bolus at t=0, integrate [0, 24] to the first decision ------
seg0    <- as.data.frame(mod |>
  init(CENT = 0) |>
  ev(amt = load_dose, cmt = 1) |>
  mrgsim(start = 0, end = cycle_h, delta = cycle_h))
st_cent <- seg0[nrow(seg0), ]$CENT   # CENT at t=24 (pre first maintenance decision)

# ---- reactive maintenance loop (carry CENT across days) -----------------------
dose <- start_dose
rows <- list()
for (k in seq_along(dec_t)) {
  t0     <- dec_t[k]
  trough <- st_cent / V           # latent signal the controller sees (pre-dose)
  if (trough < trough_lo) {
    dose   <- min(max(dose * 1.25, lo_bound), hi_bound)
    action <- "increase"
  } else if (trough > trough_hi) {
    dose   <- min(max(dose * 0.75, lo_bound), hi_bound)
    action <- "decrease"
  } else {
    action <- "reissue"
  }
  rows[[k]] <- data.frame(decision = k - 1, time = t0, trough = trough,
                          dose = dose, action = action)
  # Integrate this day's window [t0, t0 + 24] with the maintenance bolus applied at t0.
  seg     <- as.data.frame(mod |>
    init(CENT = st_cent) |>
    ev(amt = dose, cmt = 1) |>
    mrgsim(start = 0, end = cycle_h, delta = cycle_h))
  st_cent <- seg[nrow(seg), ]$CENT
}
ladder <- do.call(rbind, rows)

cum        <- sum(ladder$dose)
n_inc_rule <- sum(ladder$action == "increase")
n_inc      <- sum(diff(ladder$dose) > 0)
n_dec_a    <- sum(diff(ladder$dose) < 0)

cat("=== vancomycin loading-dose + maintenance titration (mrgsolve 1.7.2) ===\n")
cat(sprintf("loading bolus = %g mg at t=0 (pre-scheduled base regimen)\n", load_dose))
print(ladder, row.names = FALSE, digits = 10)
cat(sprintf("\ncum_maintenance_dose=%g increase_rule_fired=%d n_dose_increases=%d n_dose_decreases=%d\n",
            cum, n_inc_rule, n_inc, n_dec_a))

# ---- freeze expected.md ------------------------------------------------------
fmt <- function(x) formatC(x, format = "f", digits = 6)
ladder_tbl <- c(
  "| decision | time (h) | trough (mg/L) | maintenance dose (mg) | action |",
  "|---:|---:|---:|---:|:---|",
  apply(ladder, 1, function(r) sprintf(
    "| %d | %d | %s | %s | %s |",
    as.integer(r["decision"]), as.integer(r["time"]), fmt(as.numeric(r["trough"])),
    fmt(as.numeric(r["dose"])), trimws(r["action"]))))
md <- c(
  "# Vancomycin loading-dose + maintenance-titration reference (mrgsolve)",
  "",
  "Frozen output of `vanco_loading_mrgsolve.R` (mrgsolve 1.7.2). External anchor for",
  "the reactive `[adaptive_dosing]` **pre-scheduled base regimen** support (#702) in",
  "`examples/adaptive_vanco_loading.ferx`. NONMEM has no feedback dosing, so mrgsolve",
  "is the comparator; the R driver mirrors ferx's controller semantics exactly.",
  "Regenerate with `Rscript vanco_loading_mrgsolve.R`.",
  "",
  "1-cpt IV vancomycin. A fixed LOADING bolus rides on the subject (the base regimen);",
  "the controller then titrates DAILY MAINTENANCE boluses on the measured trough.",
  sprintf("Params (identical in the ferx model): CL=%g L/h, V=%g L. Loading bolus %g mg",
          4.0, V, load_dose),
  sprintf("at t=0. Controller: pre-dose trough (CENT/V) titrated +/-25%% to keep trough"),
  sprintf("in [%g, %g] mg/L; dose_bounds [%g, %g] mg; maintenance start_dose %g mg;",
          trough_lo, trough_hi, lo_bound, hi_bound, start_dose),
  sprintf("%d daily decisions (t = %d..%d h).", n_dec, as.integer(min(dec_t)),
          as.integer(max(dec_t))),
  "",
  "The loading dose is what makes the day-1 trough non-trivial (the decayed loading",
  "dose, not 0), so a dropped or mis-integrated base regimen would move every trough.",
  "",
  "## Realized maintenance dose ladder",
  "",
  ladder_tbl,
  "",
  "## Summary",
  "",
  sprintf("- loading_dose = %g mg (pre-scheduled base regimen, not in the reactive ledger)", load_dose),
  sprintf("- cumulative_maintenance_dose = %g mg", cum),
  sprintf("- n_increases = %d, n_decreases = %d (realized maintenance step-ups/-downs)",
          n_inc, n_dec_a),
  "",
  "ferx (`tests/adaptive_vanco_loading_anchor.rs`) reproduces this maintenance ladder",
  "dose-for-dose and the troughs to a small cross-solver tolerance (RK45 vs LSODA).",
  "The loading dose is integrated by ferx's base-regimen path (#702) and the controller",
  "titrates on top of it -- exactly what this reference exercises.")
writeLines(md, "expected.md")
cat("\nwrote expected.md\n")
