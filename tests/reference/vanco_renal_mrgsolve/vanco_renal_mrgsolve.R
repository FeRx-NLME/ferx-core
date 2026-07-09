#!/usr/bin/env Rscript
# Vancomycin trough-TDM titration under a *time-varying* renal covariate
# (epic #391 / #700).
#
# External comparator for ferx-core's reactive `[adaptive_dosing]` *continuous*
# (percentage) titration with a time-varying covariate driving PK. NONMEM has no
# native feedback dosing, so mrgsolve (which does) is the apples-to-apples anchor
# for this feature family (see docs/model-file/adaptive-dosing.qmd, "Validation").
# The ferx model under test is examples/adaptive_vanco_renal.ferx; this script
# reproduces the identical structural model + controller in mrgsolve and freezes
# the realized dose ladder to expected.md, which
# tests/adaptive_vanco_renal_anchor.rs asserts ferx matches dose-for-dose.
#
# Scenario: a 1-compartment IV vancomycin model dosed once daily by 1-h infusion.
# The patient's renal function DECLINES across the horizon — CRCL falls from ~120
# to ~40 mL/min (a realistic AKI / renal-decline trajectory) — and clearance
# scales linearly with CRCL (CL = TVCL * CRCL / 100). The controller reads the
# pre-dose trough (CENT/V) at the start of each day and titrates the dose +/-25%
# to hold the trough in [10, 15] mg/L. As CL falls, drug accumulates, so the
# controller must back the dose down — the covariate genuinely drives the ladder.
#
# --- ferx end-of-interval convention (the subtle part) ------------------------
# ferx resolves each integration segment's PK from the covariate at the record
# that ENDS the segment (NONMEM end-of-interval / $PK-runs-at-every-record), with
# LOCF carry-forward across breaks that are not data records. A decision at t_k
# splits that day into two segments, because the 1-h infusion end (t_k + 1) is a
# break that is NOT a data record:
#   * the infusion window (t_k, t_k + 1] carries the LOCF PK forward from the most
#     recent record, which after crossing the obs at t_k is CRCL(t_k);
#   * the decay window (t_k + 1, t_{k+1}] ends at the next obs record, so it is
#     governed by CRCL(t_{k+1}) (the end-of-interval value).
# So the infusion runs on CRCL AT THE DECISION time, and the between-dose decay
# runs on CRCL AT THE NEXT decision. The R loop reproduces this exactly by
# integrating each day as those two windows with the matching piecewise-constant
# CL, so mrgsolve's CL trajectory is identical to ferx's per-segment PK and the
# two engines see the same trough trajectory and reach the same dose decisions.
# (Day 0's infusion likewise uses CRCL(0): ferx seeds its LOCF carry from the
# earliest record, the t=0 obs.)
#
# The R loop mirrors ferx's controller semantics EXACTLY (confirm = 1):
#   - read the latent signal (CENT/V) at the decision time (pre-dose trough),
#   - first-matching rule wins (increase listed before decrease),
#   - `increase 25%` / `decrease 25%` scale the running dose by 1.25 / 0.75 and
#     clamp to dose_bounds (`(dose*factor).clamp(lo, hi)`),
#   - no rule match -> re-issue the running dose unchanged.
#
# Regenerate:  Rscript vanco_renal_mrgsolve.R   (run from this directory)
# Requires:    mrgsolve (tested with 1.7.2) + a C/C++ toolchain (JIT compile).

suppressMessages(library(mrgsolve))

# ---- structural model: identical params to adaptive_vanco_renal.ferx ---------
# CL is a $PARAM so it can be set per window from the active CRCL; the ferx model
# writes CL = TVCL * CRCL / 100 and resolves it per segment, which is exactly a
# piecewise-constant CL fed in at each window's end-of-interval covariate.
code <- '
$PARAM CL=4.0, V=80.0
$CMT CENT
$ODE
  dxdt_CENT = -(CL/V)*CENT;
'
mod <- mcode("vanco_renal", code, rtol = 1e-10, atol = 1e-10, maxsteps = 100000)

TVCL     <- 4.0                  # clearance at CRCL = 100 mL/min (L/h)
V        <- 80.0                 # central volume (L); CENT/V is the concentration
cycle_h  <- 24                   # once-daily dosing interval
tinf     <- 1.0                  # 1-h infusion
n_dec    <- 14                   # 14 daily decisions, t = 0 .. 312 h
dec_t    <- seq(0, by = cycle_h, length.out = n_dec)
start_dose <- 500                # empiric start (mg/day): subtherapeutic -> climbs
lo_bound  <- 250                 # dose_bounds
hi_bound  <- 4000
trough_lo <- 10                  # trough < 10 -> increase 25%
trough_hi <- 15                  # trough > 15 -> decrease 25%

# Declining renal function over the horizon (one CRCL per decision/obs time),
# CRCL(0) = 120 down to CRCL(312) = 40 mL/min. MUST match the CRCL column in
# tests/reference/vanco_renal_mrgsolve/vanco_renal_subject.csv exactly.
crcl <- c(120, 115, 108, 98, 88, 78, 68, 60, 54, 49, 45, 42, 41, 40)
stopifnot(length(crcl) == n_dec)

# CL as a function of CRCL at each decision index. cl_at[k] is the clearance
# governed by the covariate observed at decision k (used for that day's infusion
# window and, at index k+1, for the preceding decay window — see the loop below).
cl_at <- TVCL * crcl / 100

# ---- reactive loop (carry CENT across days) ----------------------------------
st_cent <- 0
dose    <- start_dose
rows    <- list()
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
  rows[[k]] <- data.frame(decision = k - 1, time = t0, crcl = crcl[k],
                          trough = trough, dose = dose, action = action)
  # Advance state across this day as TWO segments, matching ferx's per-segment PK:
  #   1. infusion window [t0, t0 + tinf] on CL from CRCL at THIS decision (LOCF),
  #   2. decay window    [t0 + tinf, t0 + 24] on CL from CRCL at the NEXT decision
  #      (end-of-interval: the next obs record governs it).
  # The last decision has no following window (nothing left to integrate).
  if (k < length(dec_t)) {
    # (1) infusion sub-window on the decision-time covariate.
    seg1 <- as.data.frame(mod |>
      param(CL = cl_at[k], V = V) |>
      init(CENT = st_cent) |>
      ev(amt = dose, rate = dose / tinf, cmt = 1) |>
      mrgsim(start = 0, end = tinf, delta = tinf))
    cent_after_inf <- seg1[nrow(seg1), ]$CENT
    # (2) decay sub-window on the next-decision (end-of-interval) covariate.
    seg2 <- as.data.frame(mod |>
      param(CL = cl_at[k + 1], V = V) |>
      init(CENT = cent_after_inf) |>
      mrgsim(start = tinf, end = cycle_h, delta = cycle_h - tinf))
    st_cent <- seg2[nrow(seg2), ]$CENT
  }
}
ladder <- do.call(rbind, rows)

cum        <- sum(ladder$dose)
n_inc_rule <- sum(ladder$action == "increase")   # times the increase RULE fired
n_dec_rule <- sum(ladder$action == "decrease")   # times the decrease RULE fired
# ferx's n_increases / n_decreases metrics count realized dose step-ups / -downs
# (consecutive ledger doses that rise / fall), NOT rule firings. Decision 0's
# increase steps off the un-realized start_dose (500 -> 625), so among the 14
# realized doses there is one fewer upward delta than increase-rule firings.
n_inc      <- sum(diff(ladder$dose) > 0)
n_dec_a    <- sum(diff(ladder$dose) < 0)
in_win     <- ladder$trough >= trough_lo & ladder$trough <= trough_hi

cat("=== vancomycin renal-decline TDM titration ladder (mrgsolve 1.7.2) ===\n")
print(ladder, row.names = FALSE, digits = 10)
cat(sprintf(
  paste0("\ncum_dose=%g increase_rule_fired=%d decrease_rule_fired=%d ",
         "n_dose_increases=%d n_dose_decreases=%d trough_in_window=%d/%d\n"),
  cum, n_inc_rule, n_dec_rule, n_inc, n_dec_a, sum(in_win), n_dec))

# ---- freeze expected.md ------------------------------------------------------
fmt <- function(x) formatC(x, format = "f", digits = 6)
ladder_tbl <- c(
  "| decision | time (h) | CRCL (mL/min) | trough (mg/L) | dose (mg) | action |",
  "|---:|---:|---:|---:|---:|:---|",
  apply(ladder, 1, function(r) sprintf(
    "| %d | %d | %s | %s | %s | %s |",
    as.integer(r["decision"]), as.integer(r["time"]), fmt(as.numeric(r["crcl"])),
    fmt(as.numeric(r["trough"])), fmt(as.numeric(r["dose"])), trimws(r["action"]))))
md <- c(
  "# Vancomycin renal-decline TDM titration reference (mrgsolve)",
  "",
  "Frozen output of `vanco_renal_mrgsolve.R` (mrgsolve 1.7.2). External anchor for",
  "the reactive `[adaptive_dosing]` *continuous* (percentage) titration with a",
  "**time-varying covariate driving PK** (#700) in",
  "`examples/adaptive_vanco_renal.ferx`. NONMEM has no feedback dosing, so mrgsolve",
  "is the comparator; the R driver mirrors ferx's controller semantics exactly and",
  "feeds CL per window from the covariate at the window's end (the NONMEM",
  "end-of-interval convention ferx's per-segment PK uses). Regenerate with",
  "`Rscript vanco_renal_mrgsolve.R`.",
  "",
  "1-cpt IV vancomycin, once-daily 1-h infusion. Params (identical in the ferx",
  sprintf("model): TVCL=%g L/h at CRCL=100, V=%g L, CL = TVCL * CRCL / 100.", TVCL, V),
  "Renal function declines over the horizon (a realistic AKI trajectory):",
  sprintf("CRCL %g -> %g mL/min, so CL falls %g -> %g L/h and drug accumulates.",
          crcl[1], crcl[n_dec], cl_at[1], cl_at[n_dec]),
  sprintf("Controller: pre-dose trough (CENT/V) titrated +/-25%% to keep trough in"),
  sprintf("[%g, %g] mg/L; dose_bounds [%g, %g] mg; start_dose %g mg; %d daily",
          trough_lo, trough_hi, lo_bound, hi_bound, start_dose, n_dec),
  sprintf("decisions (t = 0..%d h).", as.integer(max(dec_t))),
  "",
  "`auc_target` is intentionally absent: its exposure metric integrates a dense",
  "grid from a single frozen PK snapshot, which would be silently wrong under a",
  "changing covariate, so it is a typed error for a time-varying-covariate subject",
  "(#700). The pct-in-window trough metric (`target_window`) is per-event",
  "covariate-aware and retained.",
  "",
  "## Realized dose ladder",
  "",
  ladder_tbl,
  "",
  "## Summary metrics",
  "",
  sprintf("- cumulative_dose = %g mg", cum),
  sprintf("- increase rule fired %d times, decrease rule fired %d times", n_inc_rule, n_dec_rule),
  sprintf("- n_increases = %d, n_decreases = %d (realized dose step-ups/-downs; the", n_inc, n_dec_a),
  sprintf("  increase rule fired %d times, but decision 0 steps off the un-realized", n_inc_rule),
  "  start_dose, so there is one fewer realized step-up -- ferx counts dose changes)",
  sprintf("- trough_in_window = %d/%d decisions in [%g, %g] mg/L",
          sum(in_win), n_dec, trough_lo, trough_hi),
  "",
  "ferx (`tests/adaptive_vanco_renal_anchor.rs`) reproduces this ladder",
  "dose-for-dose and the troughs to a small cross-solver tolerance (RK45 vs",
  "LSODA). Both engines apply the covariate per record with the same",
  "end-of-interval convention, so the piecewise-constant CL — hence the whole",
  "trough trajectory and every dose decision — agrees.")
writeLines(md, "expected.md")
cat("\nwrote expected.md\n")
