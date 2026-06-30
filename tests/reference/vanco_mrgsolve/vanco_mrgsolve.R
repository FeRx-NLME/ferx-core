#!/usr/bin/env Rscript
# Vancomycin AUC-TDM dose-titration reference (epic #391 S2.5b).
#
# External comparator for ferx-core's reactive `[adaptive_dosing]` *continuous*
# (percentage) titration and its metrics-only AUC machinery (`auc_target` ->
# `auc_target_attainment`). NONMEM has no native feedback dosing, so mrgsolve
# (which does) is the apples-to-apples anchor for this feature family (see
# docs/model-file/adaptive-dosing.qmd, "Validation"). The ferx model under test
# is examples/adaptive_vanco_auc.ferx; this script reproduces the identical
# structural model + controller in mrgsolve and freezes the realized dose ladder,
# the per-window AUC24, and the AUC-target attainment to expected.md, which
# tests/adaptive_vanco_anchor.rs asserts ferx matches.
#
# Scenario: a 1-compartment IV vancomycin model dosed once daily by 1-h infusion.
# The controller reads the pre-dose trough (CENT/V) at the start of each day and
# titrates the dose +/-25% to keep the trough in a target band (the operational
# control law), while the *reported* outcome is AUC24 target attainment (the
# guideline endpoint, 400-600 mg*h/L) -- the real trough-vs-AUC vancomycin TDM
# tension. The exposure band never drives dosing; it only scores each day.
#
# The R loop mirrors ferx's controller semantics EXACTLY (confirm = 1):
#   - read the latent signal (CENT/V) at the decision time (pre-dose trough),
#   - first-matching rule wins (increase listed before decrease),
#   - `increase 25%` / `decrease 25%` scale the running dose by 1.25 / 0.75 and
#     clamp to dose_bounds (`(dose*factor).clamp(lo, hi)`),
#   - no rule match -> re-issue the running dose unchanged.
# AUC24 of each day is the area under CENT/V over that day, integrated by a
# dedicated AUC compartment (dxdt_AUC = CENT/V), reset at the start of each day.
#
# Regenerate:  Rscript vanco_mrgsolve.R   (run from this directory)
# Requires:    mrgsolve (tested with 1.7.2) + a C/C++ toolchain (JIT compile).

suppressMessages(library(mrgsolve))

# ---- structural model: identical params to adaptive_vanco_auc.ferx -----------
code <- '
$PARAM CL=4.0, V=80.0
$CMT CENT AUC
$ODE
  dxdt_CENT = -(CL/V)*CENT;
  dxdt_AUC  = CENT/V;            // exposure accumulator (reset each day)
'
mod <- mcode("vanco", code, rtol = 1e-10, atol = 1e-10, maxsteps = 100000)

V        <- 80.0                  # central volume (L); CENT/V is the concentration
cycle_h  <- 24                    # once-daily dosing interval
tinf      <- 1.0                  # 1-h infusion
n_dec    <- 14                    # 14 daily decisions, t = 0 .. 312 h
dec_t    <- seq(0, by = cycle_h, length.out = n_dec)
start_dose <- 500                 # empiric start (mg/day): subtherapeutic -> climbs
lo_bound  <- 250                  # dose_bounds
hi_bound  <- 4000
trough_lo <- 10                   # trough < 10 -> increase 25%
trough_hi <- 15                   # trough > 15 -> decrease 25%
auc_lo    <- 400                  # AUC24 target band (mg*h/L) -- metric only
auc_hi    <- 600

# ---- reactive loop (carry CENT across days; AUC reset within each day) --------
st_cent <- 0
dose    <- start_dose
rows    <- list()
win_auc <- numeric(0)             # one AUC24 per closed inter-decision window
for (k in seq_along(dec_t)) {
  t0     <- dec_t[k]
  trough <- st_cent / V           # latent signal the controller sees (pre-dose)
  # first-matching rule -> running dose (clamped), exactly as ferx's step_dose
  if (trough < trough_lo) {
    dose   <- min(max(dose * 1.25, lo_bound), hi_bound)
    action <- "increase"
  } else if (trough > trough_hi) {
    dose   <- min(max(dose * 0.75, lo_bound), hi_bound)
    action <- "decrease"
  } else {
    action <- "reissue"           # re-issue the running dose unchanged
  }
  rows[[k]] <- data.frame(decision = k - 1, time = t0, trough = trough,
                          dose = dose, action = action)
  # Integrate this day's window [t0, t0 + 24] with the infusion of `dose` over
  # 1 h; AUC reset to 0 so the end-of-day AUC compartment IS this day's AUC24.
  # The last decision has no following window (no AUC24 to score).
  if (k < length(dec_t)) {
    seg <- as.data.frame(mod |>
      init(CENT = st_cent, AUC = 0) |>
      ev(amt = dose, rate = dose / tinf, cmt = 1) |>
      mrgsim(start = 0, end = cycle_h, delta = cycle_h))
    last    <- seg[nrow(seg), ]
    win_auc <- c(win_auc, last$AUC)
    st_cent <- last$CENT
  }
}
ladder <- do.call(rbind, rows)

cum       <- sum(ladder$dose)
n_inc_rule <- sum(ladder$action == "increase")   # times the increase RULE fired
# ferx's n_increases metric counts realized dose step-ups (consecutive ledger
# doses that rise), NOT rule firings. They differ by one here: decision 0's
# increase steps off the un-realized start_dose (500 -> 625), so among the 14
# realized doses there are only 6 upward deltas.
n_inc     <- sum(diff(ladder$dose) > 0)
n_dec_a   <- sum(diff(ladder$dose) < 0)
in_auc    <- win_auc >= auc_lo & win_auc <= auc_hi
attain    <- mean(in_auc)         # fraction of days with AUC24 in [400, 600]

cat("=== vancomycin AUC-TDM titration ladder (mrgsolve 1.7.2) ===\n")
print(ladder, row.names = FALSE, digits = 10)
cat("\nper-window AUC24 (mg*h/L):\n")
print(round(win_auc, 6))
cat(sprintf(
  paste0("\ncum_dose=%g increase_rule_fired=%d n_dose_increases=%d n_dose_decreases=%d ",
         "auc_in_band=%d/%d attainment=%.10g\n"),
  cum, n_inc_rule, n_inc, n_dec_a, sum(in_auc), length(win_auc), attain))

# ---- freeze expected.md ------------------------------------------------------
fmt <- function(x) formatC(x, format = "f", digits = 6)
ladder_tbl <- c(
  "| decision | time (h) | trough (mg/L) | dose (mg) | action |",
  "|---:|---:|---:|---:|:---|",
  apply(ladder, 1, function(r) sprintf(
    "| %d | %d | %s | %s | %s |",
    as.integer(r["decision"]), as.integer(r["time"]), fmt(as.numeric(r["trough"])),
    fmt(as.numeric(r["dose"])), trimws(r["action"]))))
auc_tbl <- c(
  "| window | days [t0,t1] (h) | AUC24 (mg*h/L) | in [400,600] |",
  "|---:|---:|---:|:--:|",
  vapply(seq_along(win_auc), function(i) sprintf(
    "| %d | [%d, %d] | %s | %s |",
    i - 1, as.integer(dec_t[i]), as.integer(dec_t[i + 1]), fmt(win_auc[i]),
    if (in_auc[i]) "yes" else "no"), character(1)))
md <- c(
  "# Vancomycin AUC-TDM titration reference (mrgsolve)",
  "",
  "Frozen output of `vanco_mrgsolve.R` (mrgsolve 1.7.2). External anchor for the",
  "reactive `[adaptive_dosing]` *continuous* (percentage) titration and the",
  "metrics-only AUC machinery (`auc_target` -> `auc_target_attainment`) in",
  "`examples/adaptive_vanco_auc.ferx` (epic #391 S2.5b). NONMEM has no feedback",
  "dosing, so mrgsolve is the comparator; the R driver mirrors ferx's controller",
  "semantics exactly. Regenerate with `Rscript vanco_mrgsolve.R`.",
  "",
  "1-cpt IV vancomycin, once-daily 1-h infusion. Params (identical in the ferx",
  sprintf("model): CL=%g L/h, V=%g L. Controller: pre-dose trough (CENT/V) titrated",
          4.0, V),
  sprintf("+/-25%% to keep trough in [%g, %g] mg/L; dose_bounds [%g, %g] mg;",
          trough_lo, trough_hi, lo_bound, hi_bound),
  sprintf("start_dose %g mg; %d daily decisions (t = 0..%d h).", start_dose,
          n_dec, as.integer(max(dec_t))),
  "",
  sprintf("The reported outcome is AUC24 target attainment, band [%g, %g] mg*h/L",
          auc_lo, auc_hi),
  "(metric only -- it never influences dosing).",
  "",
  "## Realized dose ladder",
  "",
  ladder_tbl,
  "",
  "## Per-window AUC24 and attainment",
  "",
  auc_tbl,
  "",
  "## Summary metrics",
  "",
  sprintf("- cumulative_dose = %g mg", cum),
  sprintf("- n_increases = %d, n_decreases = %d (realized dose step-ups/-downs; the", n_inc, n_dec_a),
  sprintf("  increase rule fired %d times, but decision 0 steps off the un-realized", n_inc_rule),
  "  start_dose, so there are one fewer realized step-ups -- ferx counts dose changes)",
  sprintf("- auc_target_attainment = %d/%d = %.10g", sum(in_auc), length(win_auc), attain),
  "",
  "ferx (`tests/adaptive_vanco_anchor.rs`) reproduces this ladder dose-for-dose,",
  "the troughs to < 0.01 mg/L (a cross-solver difference), and the AUC-target",
  "attainment fraction exactly. ferx integrates the exposure by a 128-panel",
  "trapezoid per window (vs this AUC compartment), which agrees to ~1e-5",
  "relative -- far inside the margin from each AUC24 to the band edges, so the",
  "in/out classification (hence attainment) is identical.")
writeLines(md, "expected.md")
cat("\nwrote expected.md\n")
