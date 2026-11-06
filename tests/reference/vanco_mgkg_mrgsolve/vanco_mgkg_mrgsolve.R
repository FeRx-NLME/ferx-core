#!/usr/bin/env Rscript
# Weight-based (mg/kg) vancomycin titration reference (epic #391, #944).
#
# External comparator for ferx-core's per-subject dose expressions: one
# `[adaptive_dosing]` block whose `start_dose` and `dose_bounds` are expressions over
# the WT covariate, evaluated once per subject at the first decision. NONMEM has no
# native feedback dosing, so mrgsolve (which does) is the anchor for this feature
# family (docs/model-file/adaptive-dosing.qmd, "Validation").
# tests/adaptive_vanco_mgkg_anchor.rs runs examples/adaptive_vanco_mgkg.ferx through
# `simulate_adaptive_from_spec()` and asserts it matches the frozen expected.md.
#
# Scenario: 1-compartment IV vancomycin, 1-h infusions, CL = 6 * WT / 70 L/h,
# V = 0.7 * WT L. Each subject starts at 15 * WT mg; doses are clamped to
# [0, min(30 * WT, 4000)] mg. Decisions every 12 h from 0 to 144 h read the pre-dose
# trough (CENT / V) and take the first matching rule:
#   trough < 15 -> increase 25%
#   trough > 25 -> decrease 25%
# A rule acts only on its second consecutive match (confirm = 2), and the streak then
# restarts; until then, and when no rule matches, the current dose is re-issued.
#
# Subject 1 (50 kg) climbs to its 30 mg/kg cap, 1500 mg; subject 2 (150 kg) climbs to
# the 4000 mg cap, below its 30 mg/kg (4500 mg), so the `min` binds for subject 2 only
# and the two ledgers part at t = 60 h. Once at the cap an increase is absorbed: the
# rule fires and the dose stays. Every trough after t = 0 stays >= 1.3 mg/L from both
# thresholds (printed as `margin`), so no decision sits near a tie.
#
# Regenerate:  Rscript vanco_mgkg_mrgsolve.R   (run from this directory)
# Requires:    mrgsolve + a C/C++ toolchain (JIT compile).

suppressMessages(library(mrgsolve))

code <- '
$PARAM CL=6.0, V=49.0
$CMT CENT
$ODE
  dxdt_CENT = -(CL/V)*CENT;
'
mod <- mcode("vanco_mgkg", code, rtol = 1e-10, atol = 1e-10, maxsteps = 100000)

TVCL    <- 6.0
VPERKG  <- 0.7
tinf    <- 1.0
grid    <- seq(0, 144, by = 12)
th_low  <- 15
th_high <- 25
confirm <- 2
subjects <- data.frame(id = c(1, 2), wt = c(50, 150))

run_subject <- function(wt) {
  V  <- VPERKG * wt
  m  <- param(mod, CL = TVCL * wt / 70, V = V)
  lo <- 0
  hi <- min(30 * wt, 4000)
  dose <- 15 * wt
  st <- 0
  last <- NA
  streak <- 0
  rows <- list()
  for (k in seq_along(grid)) {
    t0 <- grid[k]
    trough <- st / V
    matched <- if (trough < th_low) 1 else if (trough > th_high) 2 else NA
    if (!is.na(matched) && !is.na(last) && matched == last) {
      streak <- streak + 1
    } else if (!is.na(matched)) {
      last <- matched; streak <- 1
    } else {
      last <- NA; streak <- 0
    }
    fired <- ""
    if (!is.na(matched) && streak >= confirm) {
      streak <- 0
      factor <- if (matched == 1) 1.25 else 0.75
      dose <- min(max(dose * factor, lo), hi)
      fired <- if (matched == 1) "increase" else "decrease"
    }
    margin <- if (t0 > 0) min(abs(trough - c(th_low, th_high))) else NA
    rows[[k]] <- data.frame(decision = k - 1, time = t0, trough = trough,
                            fired = fired, amt = dose, margin = margin)
    if (k < length(grid)) {
      sim <- m |> init(CENT = st) |> ev(amt = dose, rate = dose / tinf, cmt = 1)
      seg <- as.data.frame(mrgsim(sim, start = 0, end = 12, delta = 12))
      st <- seg[nrow(seg), "CENT"]
    }
  }
  do.call(rbind, rows)
}

fmt <- function(x) formatC(x, format = "f", digits = 9)
md <- c(
  "# Weight-based (mg/kg) vancomycin reference (mrgsolve)",
  "",
  sprintf("Frozen output of `vanco_mgkg_mrgsolve.R` (mrgsolve %s). External anchor",
          as.character(packageVersion("mrgsolve"))),
  "for per-subject dose expressions (#944): `start_dose = 15 * WT` and",
  "`dose_bounds = [0, min(30 * WT, 4000)]`, evaluated once per subject. NONMEM has",
  "no feedback dosing, so mrgsolve is the comparator. Regenerate with",
  "`Rscript vanco_mgkg_mrgsolve.R`.",
  "",
  "1-cpt IV vancomycin, 1-h infusions, CL = 6 * WT / 70 L/h, V = 0.7 * WT L.",
  "Decisions every 12 h, 0..144 h. Rules (first match, confirm = 2): trough < 15",
  "increase 25%; trough > 25 decrease 25%. `amt` is the dose issued at the",
  "decision; `fired` names the rule that acted; `margin` is the trough's distance to",
  "the nearer threshold.",
  "")
min_margin <- Inf
for (i in seq_len(nrow(subjects))) {
  r <- run_subject(subjects$wt[i])
  min_margin <- min(min_margin, min(r$margin, na.rm = TRUE))
  cat(sprintf("=== subject %d (WT = %g) ===\n", subjects$id[i], subjects$wt[i]))
  print(r, row.names = FALSE, digits = 10)
  md <- c(md,
    sprintf("## Subject %d (WT = %g)", subjects$id[i], subjects$wt[i]),
    "",
    "| decision | time (h) | trough (mg/L) | fired | amt (mg) | margin |",
    "|---:|---:|---:|:---|---:|---:|",
    vapply(seq_len(nrow(r)), function(j) sprintf(
      "| %d | %d | %s | %s | %s | %s |",
      as.integer(r$decision[j]), as.integer(r$time[j]), fmt(r$trough[j]),
      r$fired[j], fmt(r$amt[j]),
      if (is.na(r$margin[j])) "" else fmt(r$margin[j])), character(1)),
    "")
}
md <- c(md, sprintf("Smallest margin after t = 0: %s mg/L.", fmt(min_margin)))
cat(sprintf("\nmin margin = %g\n", min_margin))
writeLines(md, "expected.md")
cat("wrote expected.md\n")
