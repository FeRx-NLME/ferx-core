#!/usr/bin/env Rscript
# Vancomycin trough-TDM titration with INTER-OCCASION VARIABILITY on clearance
# (epic #391 / #701).
#
# External comparator for ferx-core's reactive `[adaptive_dosing]` IOV path. NONMEM
# has no native feedback dosing, so mrgsolve (which does) is the apples-to-apples
# anchor for this feature family (see docs/model-file/adaptive-dosing.qmd,
# "Validation"). The ferx model under test is examples/adaptive_vanco_iov.ferx;
# this script reproduces its structural model in mrgsolve, INJECTS the exact
# per-occasion clearance ferx used (reconstructed from the seeded κ substream, see
# below), replays ferx's realized dose ladder, and freezes the resulting trough
# trajectory to expected.md. tests/adaptive_vanco_iov_anchor.rs asserts ferx's
# live troughs match these mrgsolve troughs dose-for-dose.
#
# --- why the κ must be INJECTED (not redrawn) ---------------------------------
# ferx draws a fresh per-occasion κ_g ~ N(0, Ω_IOV) for each decision window
# (occasion = decision index, #701) on a dedicated seeded RNG substream:
#   base = subject_kappa_base_seed(seed, id, replicate); κ_g = chol(Ω_IOV)·z,
#   z = kappa_standard_normal(base, occasion=g, component=0).
# That κ is *random*, so an independent mrgsolve draw would use different numbers
# and could never match. The tractable, reliable cross-check is to reconstruct
# ferx's exact κ (the Rust reconstruction is pinned by the api.rs unit test
# `adaptive_iov_matches_predict_iov_with_reconstructed_kappa`) and feed mrgsolve
# the SAME per-occasion clearance. Then the only thing being cross-validated is the
# occasion → CL → trajectory *mechanism* — an independent ODE engine (LSODA)
# integrating the identical piecewise-constant system ferx's RK45 does — which is
# exactly the #701 surface. The κ / CL / dose constants below are ferx's, produced
# by the seed-20260708 run and reproducible from it (see the header block of
# tests/adaptive_vanco_iov_anchor.rs, which documents how to regenerate them).
#
# --- ferx end-of-interval / per-occasion convention (the subtle part) ---------
# ferx assigns each obs record to the occasion of the latest decision at-or-before
# its time (`occasion_of`), and its event-driven solver uses the end-of-interval
# (current-record) parameter convention: the integration segment ENDING at a record
# is governed by that record's CL. A decision at t_g with a 1-h infusion splits the
# day into two segments because the infusion end (t_g + 1) is a break that is NOT a
# data record:
#   * the infusion window (t_g, t_g + 1] carries LOCF PK forward from the obs at
#     t_g, which belongs to occasion g -> CL_g;
#   * the decay window (t_g + 1, t_{g+1}] ends at the next obs record (t_{g+1}),
#     which belongs to occasion g+1 -> CL_{g+1} (the end-of-interval value).
# So each day's infusion runs on THIS occasion's CL and the between-dose decay runs
# on the NEXT occasion's CL. (This is the exact per-occasion analogue of the
# renal-covariate anchor `vanco_renal_mrgsolve.R`, which does the same two-segment
# split with a covariate-driven CL instead of an occasion-driven one; empirically
# it is the only one of the three plausible conventions that reproduces ferx's
# troughs — the other two miss by ~1e-2 to ~1 mg/L.) The R loop below integrates
# each day as those two windows with the matching piecewise-constant CL, so
# mrgsolve's CL trajectory is identical to ferx's per-segment PK and the two engines
# see the same trough trajectory.
#
# Regenerate:  Rscript vanco_iov_mrgsolve.R   (run from this directory)
# Requires:    mrgsolve (tested with 1.7.2) + a C/C++ toolchain (JIT compile).

suppressMessages(library(mrgsolve))

# ---- structural model: identical params to adaptive_vanco_iov.ferx -----------
# CL is a $PARAM set per window from the injected per-occasion κ; the ferx model
# writes CL = TVCL * exp(ETA_CL + KAPPA_CL) and resolves it per segment, i.e. a
# piecewise-constant CL fed in at each window with that occasion's κ.
code <- '
$PARAM CL=4.0, V=80.0
$CMT CENT
$ODE
  dxdt_CENT = -(CL/V)*CENT;
'
mod <- mcode("vanco_iov", code, rtol = 1e-12, atol = 1e-12, maxsteps = 1000000)

TVCL     <- 4.0                  # vancomycin clearance at κ = 0 (L/h)
V        <- 80.0                 # central volume (L); CENT/V is the concentration
cycle_h  <- 24                   # once-daily dosing interval
tinf     <- 1.0                  # 1-h infusion
n_dec    <- 14                   # 14 daily decisions, t = 0 .. 312 h
dec_t    <- seq(0, by = cycle_h, length.out = n_dec)

# ---- ferx-reconstructed inputs (seed = 20260708, subject "1", replicate 1) ----
# Per-occasion κ on CL, reconstructed EXACTLY as ferx drew it (chol(Ω_IOV)·z with
# z = kappa_standard_normal(base, g, 0)); Ω_IOV = 0.09 so chol = 0.3. The api.rs
# unit test pins this reconstruction; the anchor test regenerates these live and
# asserts they equal the values frozen here (so a drift is caught, not hidden).
kappa <- c( 0.132745299519163, -0.199608898954969,  0.051584702721856,
            0.317164048542595,  0.004487691977174, -0.365058863049860,
            0.324978035712011, -0.222152229448614,  0.373939120663444,
           -0.204404711179451,  0.048066611668395, -0.231694715887291,
           -0.339563867546508, -0.179133441179247)
stopifnot(length(kappa) == n_dec)

# ferx's BSV η_CL for this subject (Ω_CL = 1e-10 -> SD 1e-5, negligible by design;
# carried so CL = TVCL * exp(η + κ) matches ferx's individual-parameter formula to
# the last ulp rather than dropping a ~1e-6 term).
eta_cl <- -0.000001382570411

# Per-occasion clearance CL_g = TVCL * exp(η + κ_g) — the piecewise-constant CL fed
# to mrgsolve. cl_at[g] governs day g's infusion window and, at index g+1, the
# preceding day's decay window (end-of-interval; see the header).
cl_at <- TVCL * exp(eta_cl + kappa)

# ferx's REALIZED dose ladder (mg/day) from the seed-20260708 reactive run. The
# controller titrated on the latent trough it observed under the occasion-varying
# CL; we replay those exact doses so mrgsolve integrates ferx's realized regimen.
dose_at <- c( 625.000000000000000,  781.250000000000000,  976.562500000000000,
             1220.703125000000000, 1525.878906250000000, 1525.878906250000000,
             1907.348632812500000, 1907.348632812500000, 2384.185791015625000,
             2384.185791015625000, 2384.185791015625000, 1788.139343261718750,
             1341.104507446289062, 1341.104507446289062)
stopifnot(length(dose_at) == n_dec)

# ---- replay loop (carry CENT across days) ------------------------------------
# Deterministic dose-for-dose replay: ferx's realized doses + reconstructed κ. Each
# day advances as two segments with the piecewise-constant CL described above; the
# pre-dose trough recorded at each decision is CENT/V at that time (occasion g's
# window has just opened, so the readout matches ferx's pre-dose observed signal).
st_cent <- 0
rows    <- list()
for (k in seq_along(dec_t)) {
  t0     <- dec_t[k]
  trough <- st_cent / V           # pre-dose trough the controller saw
  rows[[k]] <- data.frame(decision = k - 1, time = t0, kappa = kappa[k],
                          cl = cl_at[k], trough = trough, dose = dose_at[k])
  # Advance state across this day as TWO segments (matches ferx's per-segment PK):
  #   1. infusion window [t0, t0 + tinf] on THIS occasion's CL (cl_at[k]),
  #   2. decay window    [t0 + tinf, t0 + 24] on the NEXT occasion's CL (cl_at[k+1],
  #      end-of-interval). The last decision has no following window.
  if (k < length(dec_t)) {
    seg1 <- as.data.frame(mod |>
      param(CL = cl_at[k], V = V) |>
      init(CENT = st_cent) |>
      ev(amt = dose_at[k], rate = dose_at[k] / tinf, cmt = 1) |>
      mrgsim(start = 0, end = tinf, delta = tinf))
    cent_after_inf <- seg1[nrow(seg1), ]$CENT
    seg2 <- as.data.frame(mod |>
      param(CL = cl_at[k + 1], V = V) |>
      init(CENT = cent_after_inf) |>
      mrgsim(start = tinf, end = cycle_h, delta = cycle_h - tinf))
    st_cent <- seg2[nrow(seg2), ]$CENT
  }
}
ladder <- do.call(rbind, rows)

cat("=== vancomycin IOV TDM titration ladder (mrgsolve 1.7.2) ===\n")
print(ladder, row.names = FALSE, digits = 10)

# ---- freeze expected.md ------------------------------------------------------
fmt <- function(x) formatC(x, format = "f", digits = 6)
ladder_tbl <- c(
  "| decision | time (h) | kappa_CL | CL (L/h) | trough (mg/L) | dose (mg) |",
  "|---:|---:|---:|---:|---:|---:|",
  apply(ladder, 1, function(r) sprintf(
    "| %d | %d | %s | %s | %s | %s |",
    as.integer(r["decision"]), as.integer(r["time"]),
    formatC(as.numeric(r["kappa"]), format = "f", digits = 6),
    fmt(as.numeric(r["cl"])), fmt(as.numeric(r["trough"])),
    fmt(as.numeric(r["dose"])))))
md <- c(
  "# Vancomycin IOV (per-occasion κ on CL) TDM titration reference (mrgsolve)",
  "",
  "Frozen output of `vanco_iov_mrgsolve.R` (mrgsolve 1.7.2). External anchor for",
  "the reactive `[adaptive_dosing]` **inter-occasion variability** path (a fresh",
  "per-occasion κ on clearance, #701) in `examples/adaptive_vanco_iov.ferx`. NONMEM",
  "has no feedback dosing, so mrgsolve is the comparator.",
  "",
  "## Anchor form: deterministic dose-for-dose replay with reconstructed κ",
  "",
  "κ is *random* (drawn per decision window on a seeded RNG substream), so an",
  "independent mrgsolve draw could never match ferx. Instead this anchor",
  "reconstructs ferx's EXACT per-occasion κ (seed 20260708; the Rust reconstruction",
  "is pinned by the `adaptive_iov_matches_predict_iov_with_reconstructed_kappa` unit",
  "test) and injects the SAME per-occasion clearance into mrgsolve, replaying ferx's",
  "realized dose ladder. What is cross-validated is therefore the occasion → CL →",
  "trajectory *mechanism*: an independent ODE engine (LSODA) integrating the",
  "identical piecewise-constant system ferx's RK45 does. This is the #701 analogue",
  "of the #700 renal-covariate anchor — same two-segment-per-day integration, but",
  "the piecewise-constant CL comes from the per-occasion κ instead of a covariate.",
  "",
  "1-cpt IV vancomycin, once-daily 1-h infusion. Params (identical in the ferx",
  sprintf("model): TVCL=%g L/h at κ=0, V=%g L, CL = TVCL * exp(η_CL + κ_CL), Ω_IOV=0.09.", TVCL, V),
  "Each day's infusion runs on THIS occasion's CL and the decay runs on the NEXT",
  "occasion's CL (end-of-interval convention ferx's per-segment IOV PK uses). The",
  "per-occasion κ swings CL across the horizon, so the controller re-titrates each",
  "day off the trough it observes.",
  "",
  "`auc_target` is intentionally absent: its exposure metric integrates a dense grid",
  "from a single frozen PK snapshot, which is silently wrong when CL switches per",
  "occasion, so it is a typed error for an IOV (`kappa`) subject (#701). The",
  "pct-in-window trough metric (`target_window`) is per-occasion aware and retained.",
  "",
  "## Realized ladder (ferx doses + reconstructed κ, mrgsolve troughs)",
  "",
  ladder_tbl,
  "",
  "ferx (`tests/adaptive_vanco_iov_anchor.rs`) reconstructs the same per-occasion κ,",
  "computes its troughs live (RK45), and asserts they match these mrgsolve troughs",
  "(LSODA) to a small cross-solver tolerance. Both engines integrate the identical",
  "piecewise-constant CL, so the whole trough trajectory agrees.")
writeLines(md, "expected.md")
cat("\nwrote expected.md\n")
