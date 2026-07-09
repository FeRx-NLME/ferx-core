#!/usr/bin/env Rscript
# Vancomycin trough-TDM titration under BOTH a declining renal covariate AND
# inter-occasion variability on clearance (epic #391 / #700 x #701).
#
# External comparator for ferx-core's reactive `[adaptive_dosing]` COMPOSITION path.
# NONMEM has no native feedback dosing, so mrgsolve (which does) is the
# apples-to-apples anchor for this feature family (see
# docs/model-file/adaptive-dosing.qmd, "Validation"). The ferx model under test is
# examples/adaptive_vanco_renal_iov.ferx; this script reproduces its structural model
# in mrgsolve, feeds the SAME declining CRCL trajectory AND the exact per-occasion
# clearance ferx used (reconstructed from the seeded kappa substream, see below),
# replays ferx's realized dose ladder, and freezes the resulting trough trajectory to
# expected.md. tests/adaptive_vanco_renal_iov_anchor.rs asserts ferx's live troughs
# match these mrgsolve troughs dose-for-dose.
#
# --- the composition: CL = TVCL * (CRCL/100) * exp(eta + kappa) ----------------
# This is the merge of the two single-effect anchors. vanco_renal_mrgsolve.R drives
# CL from a declining CRCL; vanco_iov_mrgsolve.R drives it from a per-occasion kappa.
# Here BOTH move CL at once: each segment's CL is set by the renal function active in
# that segment AND that window's occasion kappa. The two-segment-per-day integration
# is identical to both single-effect anchors; only the piecewise-constant CL now
# folds in both effects.
#
# --- why kappa is INJECTED (not redrawn) --------------------------------------
# ferx draws a fresh per-occasion kappa_g ~ N(0, Omega_IOV) for each decision window
# (occasion = decision index, #701) on a dedicated seeded RNG substream:
#   base = subject_kappa_base_seed(seed, id, replicate); kappa_g = chol(Omega_IOV)*z,
#   z = kappa_standard_normal(base, occasion=g, component=0).
# That kappa is *random* and its substream is model-independent, so it is IDENTICAL
# to the #701 IOV anchor's kappa (same seed 20260708, subject "1", replicate 1) — the
# Rust reconstruction is pinned by the api.rs unit test
# `adaptive_iov_matches_predict_iov_with_reconstructed_kappa`. Reconstructing ferx's
# exact kappa and feeding mrgsolve the SAME per-occasion clearance makes the only
# cross-validated thing the covariate x occasion -> CL -> trajectory *mechanism* — an
# independent LSODA engine integrating the identical piecewise-constant system ferx's
# RK45 does.
#
# --- ferx end-of-interval / per-segment convention ----------------------------
# ferx uses the end-of-interval (current-record) parameter convention: the segment
# ENDING at a record is governed by that record's PK. A decision at t_g with a 1-h
# infusion splits the day into two segments:
#   * the infusion window (t_g, t_g + 1] carries LOCF PK forward from the obs at t_g,
#     which belongs to occasion g -> (CRCL_g, kappa_g);
#   * the decay window (t_g + 1, t_{g+1}] ends at the next obs record (t_{g+1}), which
#     belongs to occasion g+1 -> (CRCL_{g+1}, kappa_{g+1}) (the end-of-interval value).
# So each day's infusion runs on THIS occasion's (CRCL, kappa) and the between-dose
# decay runs on the NEXT occasion's (CRCL, kappa). The R loop below integrates each
# day as those two windows with the matching composed piecewise-constant CL.
#
# Regenerate:  Rscript vanco_renal_iov_mrgsolve.R   (run from this directory)
# Requires:    mrgsolve (tested with 1.7.2) + a C/C++ toolchain (JIT compile).

suppressMessages(library(mrgsolve))

# ---- structural model: identical params to adaptive_vanco_renal_iov.ferx ------
# CL is a $PARAM set per window from CRCL and the injected per-occasion kappa; the
# ferx model writes CL = TVCL * (CRCL/100) * exp(ETA_CL + KAPPA_CL) and resolves it
# per segment, i.e. a piecewise-constant CL fed in at each window.
code <- '
$PARAM CL=4.0, V=80.0
$CMT CENT
$ODE
  dxdt_CENT = -(CL/V)*CENT;
'
mod <- mcode("vanco_renal_iov", code, rtol = 1e-12, atol = 1e-12, maxsteps = 1000000)

TVCL     <- 4.0                  # vancomycin clearance at CRCL = 100 mL/min, kappa = 0 (L/h)
V        <- 80.0                 # central volume (L); CENT/V is the concentration
cycle_h  <- 24                   # once-daily dosing interval
tinf     <- 1.0                  # 1-h infusion
n_dec    <- 14                   # 14 daily decisions, t = 0 .. 312 h
dec_t    <- seq(0, by = cycle_h, length.out = n_dec)

# Declining renal function over the horizon (one CRCL per decision/obs time),
# CRCL(0) = 120 down to CRCL(312) = 40 mL/min. MUST match the CRCL constant in
# tests/adaptive_vanco_renal_iov_anchor.rs exactly.
crcl <- c(120, 115, 108, 98, 88, 78, 68, 60, 54, 49, 45, 42, 41, 40)
stopifnot(length(crcl) == n_dec)

# ---- ferx-reconstructed per-occasion kappa (seed 20260708, subject "1", rep 1) --
# IDENTICAL to the #701 IOV anchor's kappa (the substream is model-independent):
# chol(Omega_IOV)*z with z = kappa_standard_normal(base, g, 0); Omega_IOV = 0.09 so
# chol = 0.3. The anchor test regenerates ferx's realized ladder live at this seed and
# asserts its troughs equal the mrgsolve troughs frozen here, so a drift is caught.
kappa <- c( 0.132745299519163, -0.199608898954969,  0.051584702721856,
            0.317164048542595,  0.004487691977174, -0.365058863049860,
            0.324978035712011, -0.222152229448614,  0.373939120663444,
           -0.204404711179451,  0.048066611668395, -0.231694715887291,
           -0.339563867546508, -0.179133441179247)
stopifnot(length(kappa) == n_dec)

# ferx's BSV eta_CL for this subject (Omega_CL = 1e-10 -> SD 1e-5, negligible by
# design; carried so CL = TVCL * (CRCL/100) * exp(eta + kappa) matches ferx's
# individual-parameter formula to the last ulp).
eta_cl <- -0.000001382570411

# Composed per-occasion clearance CL_g = TVCL * (CRCL_g/100) * exp(eta + kappa_g) —
# the piecewise-constant CL fed to mrgsolve. cl_at[g] governs day g's infusion window
# and, at index g+1, the preceding day's decay window (end-of-interval; see header).
cl_at <- TVCL * (crcl / 100) * exp(eta_cl + kappa)

# ferx's REALIZED dose ladder (mg/day) from the seed-20260708 reactive run. The
# controller titrated on the latent trough it observed under the composed CL; we
# replay those exact doses so mrgsolve integrates ferx's realized regimen. Doses are
# exact binary fractions (625 * 1.25^a * 0.75^b), representable to the last ulp.
dose_at <- c( 625.000000000000000,          781.250000000000000,
              976.562500000000000,         1220.703125000000000,
             1525.878906250000000,         1525.878906250000000,
             1525.878906250000000,         1144.409179687500000,
             1144.409179687500000,          858.306884765625000,
              643.730163574218750,          482.797622680664062,
              362.098217010498047,          362.098217010498047)
stopifnot(length(dose_at) == n_dec)

# ---- replay loop (carry CENT across days) ------------------------------------
# Deterministic dose-for-dose replay: ferx's realized doses + declining CRCL +
# reconstructed kappa. Each day advances as two segments with the composed
# piecewise-constant CL; the pre-dose trough recorded at each decision is CENT/V.
st_cent <- 0
rows    <- list()
for (k in seq_along(dec_t)) {
  t0     <- dec_t[k]
  trough <- st_cent / V           # pre-dose trough the controller saw
  rows[[k]] <- data.frame(decision = k - 1, time = t0, crcl = crcl[k],
                          kappa = kappa[k], cl = cl_at[k], trough = trough,
                          dose = dose_at[k])
  # Advance state across this day as TWO segments (matches ferx's per-segment PK):
  #   1. infusion window [t0, t0 + tinf] on THIS occasion's composed CL (cl_at[k]),
  #   2. decay window    [t0 + tinf, t0 + 24] on the NEXT occasion's composed CL
  #      (cl_at[k+1], end-of-interval). The last decision has no following window.
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

cat("=== vancomycin renal-decline x IOV TDM titration ladder (mrgsolve 1.7.2) ===\n")
print(ladder, row.names = FALSE, digits = 10)

# ---- freeze expected.md ------------------------------------------------------
fmt <- function(x) formatC(x, format = "f", digits = 6)
ladder_tbl <- c(
  "| decision | time (h) | CRCL | kappa_CL | CL (L/h) | trough (mg/L) | dose (mg) |",
  "|---:|---:|---:|---:|---:|---:|---:|",
  apply(ladder, 1, function(r) sprintf(
    "| %d | %d | %d | %s | %s | %s | %s |",
    as.integer(r["decision"]), as.integer(r["time"]), as.integer(r["crcl"]),
    formatC(as.numeric(r["kappa"]), format = "f", digits = 6),
    fmt(as.numeric(r["cl"])), fmt(as.numeric(r["trough"])),
    fmt(as.numeric(r["dose"])))))
md <- c(
  "# Vancomycin renal-decline x IOV (CRCL covariate x per-occasion kappa on CL) TDM titration reference (mrgsolve)",
  "",
  "Frozen output of `vanco_renal_iov_mrgsolve.R` (mrgsolve 1.7.2). External anchor for",
  "the COMPOSITION of the reactive `[adaptive_dosing]` time-varying-covariate (#700)",
  "and inter-occasion-variability (#701) paths in `examples/adaptive_vanco_renal_iov.ferx`.",
  "NONMEM has no feedback dosing, so mrgsolve is the comparator.",
  "",
  "## Anchor form: deterministic dose-for-dose replay with declining CRCL + reconstructed kappa",
  "",
  "Clearance depends on BOTH a declining renal covariate and a per-occasion kappa:",
  sprintf("CL = TVCL * (CRCL/100) * exp(eta_CL + kappa_CL), TVCL=%g L/h, V=%g L, Omega_IOV=0.09.", TVCL, V),
  "The kappa is *random* (drawn per decision window on a seeded RNG substream) and its",
  "substream is model-independent, so it is IDENTICAL to the #701 IOV anchor's kappa",
  "(seed 20260708; the Rust reconstruction is pinned by the",
  "`adaptive_iov_matches_predict_iov_with_reconstructed_kappa` unit test). This anchor",
  "feeds mrgsolve the SAME declining CRCL AND the SAME per-occasion clearance and",
  "replays ferx's realized dose ladder, so the cross-validated thing is the",
  "covariate x occasion -> CL -> trajectory *mechanism* composed: an independent ODE",
  "engine (LSODA) integrating the identical piecewise-constant system ferx's RK45 does.",
  "",
  "This is the composition of the two single-effect anchors (renal #700, IOV #701):",
  "same two-segment-per-day integration, but the piecewise-constant CL now folds in",
  "BOTH the covariate decline and the per-occasion kappa. Each day's infusion runs on",
  "THIS occasion's (CRCL, kappa) and the decay on the NEXT occasion's (CRCL, kappa)",
  "(end-of-interval convention ferx's per-segment PK uses).",
  "",
  "`auc_target` is intentionally absent: its exposure metric integrates a dense grid",
  "from a single frozen PK snapshot, which is silently wrong when CL changes across the",
  "horizon (a drifting covariate OR a per-occasion kappa), so it is a typed error for a",
  "time-varying-covariate / IOV subject (#700/#701).",
  "",
  "## Realized ladder (ferx doses + declining CRCL + reconstructed kappa, mrgsolve troughs)",
  "",
  ladder_tbl,
  "",
  "ferx (`tests/adaptive_vanco_renal_iov_anchor.rs`) runs the reactive driver live at",
  "the same seed (declining CRCL x reconstructed kappa), computes its troughs (RK45),",
  "and asserts they match these mrgsolve troughs (LSODA) to a small cross-solver",
  "tolerance. Both engines integrate the identical composed piecewise-constant CL, so",
  "the whole trough trajectory agrees.")
writeLines(md, "expected.md")
cat("\nwrote expected.md\n")
