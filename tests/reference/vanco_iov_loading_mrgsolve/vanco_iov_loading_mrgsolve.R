#!/usr/bin/env Rscript
# Vancomycin LOADING-DOSE + maintenance-titration under INTER-OCCASION VARIABILITY
# on clearance (epic #391 / #702 / #931).
#
# External comparator for ferx-core's reactive `[adaptive_dosing]` **pre-scheduled
# base regimen under IOV** (#931): the subject starts on a fixed LOADING BOLUS (the
# #702 base regimen), a fresh per-occasion clearance is drawn for each decision window
# (occasion = decision index, #701), and the controller titrates DAILY MAINTENANCE
# boluses on the measured trough. #702/#930 rejected base × IOV; #931 lifts it — this
# run is the independent check that ferx integrates the loading dose under the
# per-occasion clearance and reaches the same ladder. NONMEM has no native feedback
# dosing, so mrgsolve (which does) is the apples-to-apples anchor for this feature
# family (see docs/model-file/adaptive-dosing.qmd, "Validation").
#
# The ferx model under test is examples/adaptive_vanco_iov_loading.ferx; the loading
# dose rides on the subject and the [adaptive_dosing] block drives the maintenance
# decisions. This script reproduces the identical structural model + loading dose +
# per-occasion CL + controller in mrgsolve and freezes the realized maintenance ladder
# (troughs + doses) to expected.md, which tests/adaptive_vanco_iov_loading_anchor.rs
# asserts ferx matches dose-for-dose.
#
# --- why the kappa must be INJECTED (not redrawn) -----------------------------
# ferx draws a fresh per-occasion kappa_g ~ N(0, Omega_IOV) for each decision window
# on a dedicated seeded RNG substream; that kappa is *random*, so an independent
# mrgsolve draw would use different numbers and could never match. The tractable
# cross-check is to reconstruct ferx's EXACT kappa (the Rust reconstruction is pinned
# by the api.rs unit test `adaptive_iov_matches_predict_iov_with_reconstructed_kappa`)
# and feed mrgsolve the SAME per-occasion clearance. Then the only thing cross-validated
# is the occasion -> CL -> trajectory *mechanism* — an independent ODE engine (LSODA)
# integrating the identical piecewise-constant system ferx's RK45 does, with a base
# loading dose riding on top — which is exactly the #931 surface. The kappa / eta
# constants below are ferx's, produced by the seed-20260726 run and reproducible from it
# (see the header of tests/adaptive_vanco_iov_loading_anchor.rs, which documents how to
# regenerate them via the in-crate harness).
#
# --- ferx end-of-interval / per-occasion convention (the subtle part) ---------
# ferx resolves each integration segment's PK from the record that ENDS the segment
# (NONMEM end-of-interval), and assigns each record to the occasion of the latest
# decision at-or-before its time. Every dose here is an instantaneous BOLUS, so each day
# is a SINGLE segment:
#   * the LOADING bolus at t=0 decays over (0, 24] ending at the record at t=24, which
#     belongs to occasion 0 -> CL_0 (NOT the baseline kappa=0 window, even though the
#     dose is administered at t=0);
#   * a maintenance bolus at decision t_k = 24k decays over (t_k, t_k + 24] ending at the
#     record at t=24(k+1), which belongs to occasion k+1 -> CL_{k+1}.
# The R loop reproduces this by decaying each day under the NEXT record's occasion CL, so
# mrgsolve's piecewise-constant CL matches ferx's per-segment PK and the two engines see
# the same trough trajectory and reach the same dose decisions.
#
# The R loop mirrors ferx's controller semantics EXACTLY (confirm = 1):
#   - read the latent signal (CENT/V) at the decision time (pre-dose trough),
#   - first-matching rule wins (increase listed before decrease),
#   - `increase 25%` / `decrease 25%` scale the running dose by 1.25 / 0.75 and clamp to
#     dose_bounds, no rule match -> re-issue the running dose unchanged.
#
# Regenerate:  Rscript vanco_iov_loading_mrgsolve.R   (run from this directory)
# Requires:    mrgsolve (tested with 1.7.2) + a C/C++ toolchain (JIT compile).

suppressMessages(library(mrgsolve))

# ---- structural model: identical params to adaptive_vanco_iov_loading.ferx ----
# CL is a $PARAM so it can be set per day from the active per-occasion kappa; the ferx
# model writes CL = TVCL * exp(ETA_CL + KAPPA_CL) and resolves it per segment.
code <- '
$PARAM CL=4.0, V=80.0
$CMT CENT
$ODE
  dxdt_CENT = -(CL/V)*CENT;
'
mod <- mcode("vanco_iov_loading", code, rtol = 1e-10, atol = 1e-10, maxsteps = 100000)

TVCL       <- 4.0                 # vancomycin clearance at kappa = 0 (L/h)
V          <- 80.0                # central volume (L); CENT/V is the concentration
cycle_h    <- 24                  # once-daily maintenance interval
load_dose  <- 1500               # pre-scheduled LOADING bolus at t=0 (#702 base regimen)
n_dec      <- 8                   # 8 daily maintenance decisions, t = 24 .. 192 h
dec_t      <- seq(cycle_h, by = cycle_h, length.out = n_dec)  # 24, 48, ..., 192
start_dose <- 750                 # empiric maintenance start (mg/day)
lo_bound   <- 250                 # dose_bounds
hi_bound   <- 4000
trough_lo  <- 10                  # trough < 10 -> increase 25%
trough_hi  <- 15                  # trough > 15 -> decrease 25%

# ---- ferx-reconstructed inputs (seed = 20260726, subject "1", replicate 1) ----
# Per-occasion kappa on CL, reconstructed EXACTLY as ferx drew it (chol(Omega_IOV)*z,
# z = kappa_standard_normal(base, g, 0)); Omega_IOV = 0.09 so chol = 0.3. One kappa per
# decision window (occasion 0..7). The negligible BSV eta_CL (Omega_CL = 1e-10 -> SD 1e-5)
# is carried so CL = TVCL * exp(eta + kappa) matches ferx's formula to the last ulp.
eta_cl <- 0.00000088583491527
kappa  <- c(-0.17452934778206788, -0.10348649992799583,  0.26989460564368989,
            -0.32334615885867241, -0.75545404787741377,  0.12037108591248322,
            -0.31657414773672443, -0.13157668289686250)
stopifnot(length(kappa) == n_dec)

# cl_at[j] governs the segment ENDING at record j (t = 0, 24, ..., 192; 9 records).
# Record 1 (t=0) is the baseline window (kappa=0) and is never a decay-into target (the
# loading dose is applied there); records 2..9 (t=24..192) belong to occasions 0..7.
cl_at <- c(TVCL * exp(eta_cl), TVCL * exp(eta_cl + kappa))   # length n_dec + 1 = 9
stopifnot(length(cl_at) == n_dec + 1)

bolus_day <- function(cent0, amt, cl) {
  seg <- as.data.frame(mod |>
    param(CL = cl, V = V) |>
    init(CENT = cent0) |>
    ev(amt = amt, cmt = 1) |>
    mrgsim(start = 0, end = cycle_h, delta = cycle_h))
  seg[nrow(seg), ]$CENT
}

# ---- loading bolus at t=0, decay (0, 24] under occasion 0 = cl_at[2] ----------
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
  # CL of the occasion this decision opens (occasion k-1, i.e. cl_at[k+1]) — reported for
  # the ladder table; the decay below uses the NEXT record's occasion CL (end-of-interval).
  rows[[k]] <- data.frame(decision = k - 1, time = t0, cl = cl_at[k + 1],
                          trough = trough, dose = dose, action = action)
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

cat("=== vancomycin per-occasion IOV LOADING + maintenance titration (mrgsolve 1.7.2) ===\n")
cat(sprintf("loading bolus = %g mg at t=0 (pre-scheduled base regimen, under per-occasion CL)\n",
            load_dose))
print(ladder, row.names = FALSE, digits = 10)
cat(sprintf(paste0("\ncum_maintenance_dose=%g increase_rule_fired=%d decrease_rule_fired=%d ",
                   "n_dose_increases=%d n_dose_decreases=%d trough_in_window=%d/%d\n"),
            cum, n_inc_rule, n_dec_rule, n_inc, n_dec_a, sum(in_win), n_dec))

# ---- freeze expected.md ------------------------------------------------------
fmt <- function(x) formatC(x, format = "f", digits = 6)
ladder_tbl <- c(
  "| decision | time (h) | CL (L/h) | trough (mg/L) | maintenance dose (mg) | action |",
  "|---:|---:|---:|---:|---:|:---|",
  apply(ladder, 1, function(r) sprintf(
    "| %d | %d | %s | %s | %s | %s |",
    as.integer(r["decision"]), as.integer(r["time"]), fmt(as.numeric(r["cl"])),
    fmt(as.numeric(r["trough"])), fmt(as.numeric(r["dose"])), trimws(r["action"]))))
md <- c(
  "# Vancomycin per-occasion IOV LOADING + maintenance-titration reference (mrgsolve)",
  "",
  "Frozen output of `vanco_iov_loading_mrgsolve.R` (mrgsolve 1.7.2). External anchor for",
  "the reactive `[adaptive_dosing]` **pre-scheduled base regimen under inter-occasion",
  "variability** (#931) in `examples/adaptive_vanco_iov_loading.ferx`. NONMEM has no feedback",
  "dosing, so mrgsolve is the comparator; the R driver mirrors ferx's controller semantics",
  "exactly and feeds CL per day from the occasion at the segment's end (the NONMEM",
  "end-of-interval convention ferx's per-segment PK uses). Regenerate with",
  "`Rscript vanco_iov_loading_mrgsolve.R`.",
  "",
  "1-cpt IV vancomycin, bolus dosing. A fixed LOADING bolus rides on the subject (the base",
  "regimen); the controller then titrates DAILY MAINTENANCE boluses on the measured trough,",
  "while a fresh per-occasion clearance is drawn each decision window. Params (identical in",
  "the ferx model):",
  sprintf("TVCL=%g L/h at kappa=0, V=%g L, CL = TVCL * exp(eta + kappa). Loading bolus %g mg at t=0.",
          TVCL, V, load_dose),
  sprintf("Per-occasion kappa swings CL %g -> %g L/h across the horizon (occasions 0..7).",
          min(cl_at[-1]), max(cl_at[-1])),
  sprintf("Controller: pre-dose trough (CENT/V) titrated +/-25%% to keep trough in [%g, %g]",
          trough_lo, trough_hi),
  sprintf("mg/L; dose_bounds [%g, %g] mg; maintenance start_dose %g mg; %d daily decisions",
          lo_bound, hi_bound, start_dose, n_dec),
  sprintf("(t = %d..%d h).", as.integer(min(dec_t)), as.integer(max(dec_t))),
  "",
  "The loading dose decayed under occasion 0's clearance (the end-of-interval value for the",
  "(0, 24] segment) is what makes the day-1 trough non-trivial (a dropped base regimen, or one",
  "frozen at kappa=0, would move every trough). This is the base-regimen path (#702) combined",
  "with per-occasion IOV PK (#701).",
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
  "ferx (`tests/adaptive_vanco_iov_loading_anchor.rs`) reproduces this maintenance ladder",
  "dose-for-dose and the troughs to a small cross-solver tolerance (RK45 vs LSODA). The",
  "loading dose is integrated by ferx's base-regimen path under the per-occasion clearance",
  "(#931), and the controller titrates on top of it -- exactly what this reference exercises.")
writeLines(md, "expected.md")
cat("\nwrote expected.md\n")
