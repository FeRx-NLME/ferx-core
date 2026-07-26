#!/usr/bin/env Rscript
# Vancomycin LOADING-DOSE + maintenance-titration under a *time-varying* renal
# covariate (epic #391 / #702 / #930).
#
# External comparator for ferx-core's reactive `[adaptive_dosing]` **pre-scheduled
# base regimen under a time-varying covariate** (#930): the subject starts on a fixed
# LOADING BOLUS (the #702 base regimen), the patient's renal function DECLINES over the
# horizon (CRCL falls, so CL falls, #700), and the controller titrates DAILY MAINTENANCE
# boluses on the measured trough. #702 rejected base × time-varying covariate; #930 lifts
# it — this run is the independent check that ferx integrates the loading dose under the
# declining per-segment clearance and reaches the same ladder. NONMEM has no native
# feedback dosing, so mrgsolve (which does) is the apples-to-apples anchor for this feature
# family (see docs/model-file/adaptive-dosing.qmd, "Validation").
#
# The ferx model under test is examples/adaptive_vanco_renal_loading.ferx; the loading dose
# rides on the subject (vanco_renal_loading_subject.csv) and the [adaptive_dosing] block
# drives the maintenance decisions. This script reproduces the identical structural model +
# loading dose + declining CL + controller in mrgsolve and freezes the realized maintenance
# ladder (troughs + doses) to expected.md, which tests/adaptive_vanco_renal_loading_anchor.rs
# asserts ferx matches dose-for-dose.
#
# --- ferx end-of-interval convention (the subtle part, bolus edition) ---------
# ferx resolves each integration segment's PK from the covariate at the record that ENDS
# the segment (NONMEM end-of-interval), with LOCF carry-forward. Every dose here is an
# instantaneous BOLUS, so each day is a SINGLE segment (no infusion sub-window): the segment
# (t_k, t_k + 24] ends at the next trough record t_{k+1}, so it decays under CRCL(t_{k+1}).
#   * the loading bolus at t=0 decays over (0, 24] under CRCL(24) (the first trough record);
#   * a maintenance bolus at decision t_k decays over (t_k, t_k + 24] under CRCL(t_{k+1}).
# The R loop reproduces this exactly by decaying each day under the NEXT record's CRCL, so
# mrgsolve's piecewise-constant CL matches ferx's per-segment PK and the two engines see the
# same trough trajectory and reach the same dose decisions.
#
# The R loop mirrors ferx's controller semantics EXACTLY (confirm = 1):
#   - read the latent signal (CENT/V) at the decision time (pre-dose trough),
#   - first-matching rule wins (increase listed before decrease),
#   - `increase 25%` / `decrease 25%` scale the running dose by 1.25 / 0.75 and clamp to
#     dose_bounds, no rule match -> re-issue the running dose unchanged.
#
# Regenerate:  Rscript vanco_renal_loading_mrgsolve.R   (run from this directory)
# Requires:    mrgsolve (tested with 1.7.2) + a C/C++ toolchain (JIT compile).

suppressMessages(library(mrgsolve))

# ---- structural model: identical params to adaptive_vanco_renal_loading.ferx -
# CL is a $PARAM so it can be set per day from the active CRCL; the ferx model writes
# CL = TVCL * CRCL / 100 and resolves it per segment (piecewise-constant CL).
code <- '
$PARAM CL=4.0, V=80.0
$CMT CENT
$ODE
  dxdt_CENT = -(CL/V)*CENT;
'
mod <- mcode("vanco_renal_loading", code, rtol = 1e-10, atol = 1e-10, maxsteps = 100000)

TVCL       <- 4.0                 # clearance at CRCL = 100 mL/min (L/h)
V          <- 80.0                # central volume (L); CENT/V is the concentration
cycle_h    <- 24                  # once-daily maintenance interval
load_dose  <- 1500                # pre-scheduled LOADING bolus at t=0 (#702 base regimen)
n_dec      <- 8                   # 8 daily maintenance decisions, t = 24 .. 192 h
dec_t      <- seq(cycle_h, by = cycle_h, length.out = n_dec)  # 24, 48, ..., 192
start_dose <- 750                 # empiric maintenance start (mg/day)
lo_bound   <- 250                 # dose_bounds
hi_bound   <- 4000
trough_lo  <- 10                  # trough < 10 -> increase 25%
trough_hi  <- 15                  # trough > 15 -> decrease 25%

# Declining renal function (one CRCL per record time t = 0, 24, ..., 192), CRCL(0)=120
# down to CRCL(192)=41 mL/min. MUST match the CRCL column in
# tests/reference/vanco_renal_loading_mrgsolve/vanco_renal_loading_subject.csv exactly.
crcl <- c(120, 100, 85, 72, 62, 54, 48, 44, 41)
stopifnot(length(crcl) == n_dec + 1)         # t = 0 plus the n_dec decision times
cl_at <- TVCL * crcl / 100                    # cl_at[j] governs the segment ENDING at record j

bolus_day <- function(cent0, amt, cl) {
  seg <- as.data.frame(mod |>
    param(CL = cl, V = V) |>
    init(CENT = cent0) |>
    ev(amt = amt, cmt = 1) |>
    mrgsim(start = 0, end = cycle_h, delta = cycle_h))
  seg[nrow(seg), ]$CENT
}

# ---- loading bolus at t=0, decay (0, 24] under CRCL(24) = cl_at[2] ------------
st_cent <- bolus_day(0, load_dose, cl_at[2])   # CENT at t=24 (pre first maintenance decision)

# ---- reactive maintenance loop (carry CENT across days) ----------------------
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
    action <- "reissue"           # re-issue the running dose unchanged
  }
  rows[[k]] <- data.frame(decision = k - 1, time = t0, crcl = crcl[k + 1],
                          trough = trough, dose = dose, action = action)
  # Apply the maintenance bolus at t0 and decay (t0, t0 + 24] under the NEXT record's CRCL
  # (end-of-interval). The last decision has no following window (horizon ends at t0).
  if (k < length(dec_t)) {
    st_cent <- bolus_day(st_cent, dose, cl_at[k + 2])
  }
}
ladder <- do.call(rbind, rows)

cum        <- sum(ladder$dose)
n_inc_rule <- sum(ladder$action == "increase")
n_dec_rule <- sum(ladder$action == "decrease")
n_inc      <- sum(diff(ladder$dose) > 0)
n_dec_a    <- sum(diff(ladder$dose) < 0)
in_win     <- ladder$trough >= trough_lo & ladder$trough <= trough_hi

cat("=== vancomycin renal-decline LOADING + maintenance titration (mrgsolve 1.7.2) ===\n")
cat(sprintf("loading bolus = %g mg at t=0 (pre-scheduled base regimen, under declining CL)\n",
            load_dose))
print(ladder, row.names = FALSE, digits = 10)
cat(sprintf(paste0("\ncum_maintenance_dose=%g increase_rule_fired=%d decrease_rule_fired=%d ",
                   "n_dose_increases=%d n_dose_decreases=%d trough_in_window=%d/%d\n"),
            cum, n_inc_rule, n_dec_rule, n_inc, n_dec_a, sum(in_win), n_dec))

# ---- freeze expected.md ------------------------------------------------------
fmt <- function(x) formatC(x, format = "f", digits = 6)
ladder_tbl <- c(
  "| decision | time (h) | CRCL (mL/min) | trough (mg/L) | maintenance dose (mg) | action |",
  "|---:|---:|---:|---:|---:|:---|",
  apply(ladder, 1, function(r) sprintf(
    "| %d | %d | %s | %s | %s | %s |",
    as.integer(r["decision"]), as.integer(r["time"]), fmt(as.numeric(r["crcl"])),
    fmt(as.numeric(r["trough"])), fmt(as.numeric(r["dose"])), trimws(r["action"]))))
md <- c(
  "# Vancomycin renal-decline LOADING + maintenance-titration reference (mrgsolve)",
  "",
  "Frozen output of `vanco_renal_loading_mrgsolve.R` (mrgsolve 1.7.2). External anchor for",
  "the reactive `[adaptive_dosing]` **pre-scheduled base regimen under a time-varying",
  "covariate** (#930) in `examples/adaptive_vanco_renal_loading.ferx`. NONMEM has no feedback",
  "dosing, so mrgsolve is the comparator; the R driver mirrors ferx's controller semantics",
  "exactly and feeds CL per day from the covariate at the segment's end (the NONMEM",
  "end-of-interval convention ferx's per-segment PK uses). Regenerate with",
  "`Rscript vanco_renal_loading_mrgsolve.R`.",
  "",
  "1-cpt IV vancomycin, bolus dosing. A fixed LOADING bolus rides on the subject (the base",
  "regimen); the controller then titrates DAILY MAINTENANCE boluses on the measured trough,",
  "while renal function declines. Params (identical in the ferx model):",
  sprintf("TVCL=%g L/h at CRCL=100, V=%g L, CL = TVCL * CRCL / 100. Loading bolus %g mg at t=0.",
          TVCL, V, load_dose),
  sprintf("Renal function declines CRCL %g -> %g mL/min, so CL falls %g -> %g L/h and drug",
          crcl[1], crcl[n_dec + 1], cl_at[1], cl_at[n_dec + 1]),
  "accumulates.",
  sprintf("Controller: pre-dose trough (CENT/V) titrated +/-25%% to keep trough in [%g, %g]",
          trough_lo, trough_hi),
  sprintf("mg/L; dose_bounds [%g, %g] mg; maintenance start_dose %g mg; %d daily decisions",
          lo_bound, hi_bound, start_dose, n_dec),
  sprintf("(t = %d..%d h).", as.integer(min(dec_t)), as.integer(max(dec_t))),
  "",
  "The loading dose decayed under the DECLINING clearance is what makes the day-1 trough",
  "non-trivial (a dropped base regimen, or one frozen at the t=0 covariate, would move every",
  "trough). This is the base-regimen path (#702) combined with per-event covariate PK (#700).",
  "",
  "## Realized maintenance dose ladder",
  "",
  ladder_tbl,
  "",
  "## Summary",
  "",
  sprintf("- loading_dose = %g mg (pre-scheduled base regimen, not in the reactive ledger)",
          load_dose),
  sprintf("- cumulative_maintenance_dose = %g mg", cum),
  sprintf("- increase rule fired %d times, decrease rule fired %d times", n_inc_rule, n_dec_rule),
  sprintf("- n_increases = %d, n_decreases = %d (realized maintenance step-ups/-downs)",
          n_inc, n_dec_a),
  sprintf("- trough_in_window = %d/%d decisions in [%g, %g] mg/L",
          sum(in_win), n_dec, trough_lo, trough_hi),
  "",
  "ferx (`tests/adaptive_vanco_renal_loading_anchor.rs`) reproduces this maintenance ladder",
  "dose-for-dose and the troughs to a small cross-solver tolerance (RK45 vs LSODA). The",
  "loading dose is integrated by ferx's base-regimen path under the declining per-segment",
  "clearance (#930), and the controller titrates on top of it -- exactly what this reference",
  "exercises.")
writeLines(md, "expected.md")
cat("\nwrote expected.md\n")
