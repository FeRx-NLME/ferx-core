#!/usr/bin/env Rscript
# Vancomycin interval-extension reference (epic #391, #945).
#
# External comparator for ferx-core's schedule adaptation: the reactive driver
# keeps a fixed review grid and consults the controller only at grid points that
# are due under the current dosing interval, and the controller can lengthen or
# shorten that interval. NONMEM has no native feedback dosing, so mrgsolve (which
# does) is the anchor for this feature family (docs/model-file/adaptive-dosing.qmd,
# "Validation"). tests/adaptive_vanco_interval_anchor.rs runs the identical model
# and controller through ferx's `simulate_adaptive()` and asserts it matches the
# frozen expected.md.
#
# Scenario: 1-compartment IV vancomycin, a fixed 1000 mg dose by 1-h infusion,
# CL = 4 * RF L/h (RF = renal fraction, constant per subject), V = 80 L. The
# review grid is every 12 h from 0 to 288 h; the dosing interval moves on the
# ladder 12 / 24 / 36 / 48 h, starting at 12 h. At a due grid point the
# controller reads the pre-dose trough (CENT/V) and takes the first matching rule:
#   trough > 25   -> hold, lengthen
#   trough > 14   -> dose, lengthen
#   trough < 9    -> dose, shorten
#   otherwise     -> dose
# lengthen / shorten step one rung and saturate at both ends. A consulted
# decision (dose or hold) starts the next interval: the next grid point due is
# t + interval. Grid points that are not due read the trough and are logged as
# not_due; nothing is dosed there.
#
# Subject 1 (RF = 0.15) accumulates, lengthens to the top rung, stays pinned there
# and holds at due points (one of them on the top rung). Subject 2 (RF = 0.5)
# lengthens twice, shortens, and lengthens again. Decision 0 of both shortens at
# the bottom rung (trough 0 < 9), so the ladder saturates at both ends. Every
# consulted trough stays >= 0.13 mg/L from its nearest threshold (printed as
# `margin`), ~1000x the cross-solver trough gap, so no decision sits on a tie.
#
# Regenerate:  Rscript vanco_interval_mrgsolve.R   (run from this directory)
# Requires:    mrgsolve + a C/C++ toolchain (JIT compile).

suppressMessages(library(mrgsolve))

code <- '
$PARAM CL=4.0, V=80.0
$CMT CENT
$ODE
  dxdt_CENT = -(CL/V)*CENT;
'
mod <- mcode("vanco_interval", code, rtol = 1e-10, atol = 1e-10, maxsteps = 100000)

V      <- 80.0
TVCL   <- 4.0
dose   <- 1000
tinf   <- 1.0
grid   <- seq(0, 288, by = 12)
ladder <- c(12, 24, 36, 48)
start_rung <- 1
th_hold <- 25
th_len  <- 14
th_short <- 9
subjects <- data.frame(id = c(1, 2), rf = c(0.15, 0.5))

run_subject <- function(rf) {
  m <- param(mod, CL = TVCL * rf)
  st <- 0
  rung <- start_rung
  next_due <- NA
  rows <- list()
  for (k in seq_along(grid)) {
    t0 <- grid[k]
    trough <- st / V
    due <- is.na(next_due) || t0 >= next_due - 1e-9
    give <- FALSE
    if (!due) {
      outcome <- "not_due"
    } else {
      if (trough > th_hold) {
        outcome <- "hold"; rung <- min(rung + 1, length(ladder))
      } else if (trough > th_len) {
        outcome <- "dosed"; give <- TRUE; rung <- min(rung + 1, length(ladder))
      } else if (trough < th_short) {
        outcome <- "dosed"; give <- TRUE; rung <- max(rung - 1, 1)
      } else {
        outcome <- "dosed"; give <- TRUE
      }
      next_due <- t0 + ladder[rung]
    }
    margin <- if (due) min(abs(trough - c(th_hold, th_len, th_short))) else NA
    rows[[k]] <- data.frame(decision = k - 1, time = t0, trough = trough,
                            outcome = outcome, interval = ladder[rung],
                            margin = margin)
    if (k < length(grid)) {
      sim <- m |> init(CENT = st)
      if (give) sim <- sim |> ev(amt = dose, rate = dose / tinf, cmt = 1)
      seg <- as.data.frame(mrgsim(sim, start = 0, end = 12, delta = 12))
      st <- seg[nrow(seg), "CENT"]
    }
  }
  do.call(rbind, rows)
}

# 9 decimals: the ferx anchor agrees to ~1e-7 with eta = 0, so 6 decimals would make
# the frozen table, not the engines, the limit of the comparison.
fmt <- function(x) formatC(x, format = "f", digits = 9)
md <- c(
  "# Vancomycin interval-extension reference (mrgsolve)",
  "",
  sprintf("Frozen output of `vanco_interval_mrgsolve.R` (mrgsolve %s). External",
          as.character(packageVersion("mrgsolve"))),
  "anchor for schedule adaptation (#945): a fixed review grid, a due mask, and an",
  "interval ladder moved by the controller. NONMEM has no feedback dosing, so",
  "mrgsolve is the comparator. Regenerate with `Rscript vanco_interval_mrgsolve.R`.",
  "",
  "1-cpt IV vancomycin, 1000 mg by 1-h infusion, CL = 4 * RF L/h, V = 80 L.",
  "Review grid every 12 h, 0..288 h. Interval ladder 12/24/36/48 h, start 12 h.",
  "Due-point rules (first match): trough > 25 hold + lengthen; > 14 dose +",
  "lengthen; < 9 dose + shorten; else dose. `margin` is the trough's distance to",
  "the nearest threshold at a consulted decision.",
  "")
min_margin <- Inf
for (i in seq_len(nrow(subjects))) {
  r <- run_subject(subjects$rf[i])
  min_margin <- min(min_margin, min(r$margin, na.rm = TRUE))
  cat(sprintf("=== subject %d (RF = %g) ===\n", subjects$id[i], subjects$rf[i]))
  print(r, row.names = FALSE, digits = 10)
  md <- c(md,
    sprintf("## Subject %d (RF = %g)", subjects$id[i], subjects$rf[i]),
    "",
    "| decision | time (h) | trough (mg/L) | outcome | interval after (h) | margin |",
    "|---:|---:|---:|:---|---:|---:|",
    vapply(seq_len(nrow(r)), function(j) sprintf(
      "| %d | %d | %s | %s | %d | %s |",
      as.integer(r$decision[j]), as.integer(r$time[j]), fmt(r$trough[j]),
      r$outcome[j], as.integer(r$interval[j]),
      if (is.na(r$margin[j])) "" else fmt(r$margin[j])), character(1)),
    "")
}
md <- c(md, sprintf("Smallest margin at a consulted decision: %s mg/L.", fmt(min_margin)))
cat(sprintf("\nmin margin = %g\n", min_margin))
writeLines(md, "expected.md")
cat("wrote expected.md\n")
