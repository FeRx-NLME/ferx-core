#!/usr/bin/env Rscript
# Weight-based (mg/kg) vancomycin titration reference (epic #391, #944).
#
# External comparator for ferx-core's per-subject dose expressions: one
# `[adaptive_dosing]` block whose `start_dose` and `dose_bounds` are expressions over
# the WT covariate, evaluated once per subject at the first decision. NONMEM has no
# native feedback dosing, so mrgsolve (which does) is the anchor for this feature
# family (docs/model-file/adaptive-dosing.qmd, "Validation").
# tests/adaptive_vanco_mgkg_anchor.rs runs examples/adaptive_vanco_mgkg.ferx and
# examples/adaptive_vanco_mgkg_rounded.ferx through `simulate_adaptive_from_spec()`
# and asserts each matches its frozen table.
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
# The `round_to` arm (#944 PR3, expected_round_to.md, vanco_mgkg_round_subjects.csv)
# runs the same rules with `round_to = 250` and `dose_bounds = [250, min(30 * WT,
# 4000)]`: every dose (the start dose too) is rounded to the nearest multiple of 250,
# ties up, then clamped to the multiples inside the bounds, and the rounded dose is
# carried. Subjects 40 kg and 80 kg. Between them:
#   - the start dose is rounded (600 -> 500, 1200 -> 1250);
#   - a tie goes up (625 -> 750 at 40 kg, 1875 -> 2000 at 80 kg);
#   - the cap needs aligning (1200 -> 1000, 2400 -> 2250), and clamping before
#     rounding would give a dose above it (1250 > 1200, 2500 > 2400);
#   - the carried dose matters: at 40 kg the second step starts from 750 and gives
#     1000, where carrying the unrounded 625 would give 781.25 -> 750.
# The `raw` column is the stepped dose before rounding.
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

# Nearest multiple of s, ties up: k = floor(x/s + 0.5 + 1e-9 * max(|x/s|, 1)).
round_grid <- function(x, s) {
  q <- x / s
  floor(q + 0.5 + 1e-9 * max(abs(q), 1)) * s
}
# ceil(lo/s) and floor(hi/s), snapping a quotient within 1e-9 (relative) of an integer.
snap <- function(q, f) {
  r <- round(q)
  if (abs(q - r) <= 1e-9 * max(abs(r), 1)) r else f(q)
}

run_subject <- function(wt, lo = 0, step = NA) {
  V  <- VPERKG * wt
  m  <- param(mod, CL = TVCL * wt / 70, V = V)
  hi <- min(30 * wt, 4000)
  place <- if (is.na(step)) {
    function(x) min(max(x, lo), hi)
  } else {
    alo <- snap(lo / step, ceiling) * step
    ahi <- snap(hi / step, floor) * step
    function(x) min(max(round_grid(x, step), alo), ahi)
  }
  dose <- place(15 * wt)
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
    raw <- if (k == 1) 15 * wt else NA
    if (!is.na(matched) && streak >= confirm) {
      streak <- 0
      factor <- if (matched == 1) 1.25 else 0.75
      raw <- dose * factor
      dose <- place(raw)
      fired <- if (matched == 1) "increase" else "decrease"
    }
    margin <- if (t0 > 0) min(abs(trough - c(th_low, th_high))) else NA
    rows[[k]] <- data.frame(decision = k - 1, time = t0, trough = trough,
                            fired = fired, raw = raw, amt = dose, margin = margin)
    if (k < length(grid)) {
      sim <- m |> init(CENT = st) |> ev(amt = dose, rate = dose / tinf, cmt = 1)
      seg <- as.data.frame(mrgsim(sim, start = 0, end = 12, delta = 12))
      st <- seg[nrow(seg), "CENT"]
    }
  }
  do.call(rbind, rows)
}

fmt <- function(x) formatC(x, format = "f", digits = 9)

# The original arm: expected.md, byte-for-byte as before PR3 (no `raw` column).
write_plain <- function(subjects, path) {
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
  writeLines(md, path)
  cat(sprintf("wrote %s\n", path))
}

# The `round_to = 250` arm (#944 PR3): the same table plus `raw`, the dose before it
# went onto the grid (the start dose, or the stepped dose).
write_rounded <- function(subjects, path, lo, step) {
  md <- c(
    "# Weight-based (mg/kg) vancomycin reference, rounded doses (mrgsolve)",
    "",
    sprintf("Frozen output of `vanco_mgkg_mrgsolve.R` (mrgsolve %s), the `round_to` arm.",
            as.character(packageVersion("mrgsolve"))),
    "External anchor for dose rounding (#944 PR3): `start_dose = 15 * WT`,",
    sprintf("`dose_bounds = [%g, min(30 * WT, 4000)]`, `round_to = %g`. Every dose is", lo, step),
    "rounded to the nearest multiple (ties up), then clamped to the multiples inside",
    "the bounds, and the rounded dose is carried. Regenerate with",
    "`Rscript vanco_mgkg_mrgsolve.R`.",
    "",
    "Model and rules as in `expected.md`. `raw` is the dose before rounding (the",
    "start dose at decision 0, the stepped dose when a rule fired); `amt` is the dose",
    "issued.",
    "")
  min_margin <- Inf
  for (i in seq_len(nrow(subjects))) {
    r <- run_subject(subjects$wt[i], lo = lo, step = step)
    min_margin <- min(min_margin, min(r$margin, na.rm = TRUE))
    cat(sprintf("=== round_to subject %d (WT = %g) ===\n", subjects$id[i], subjects$wt[i]))
    print(r, row.names = FALSE, digits = 10)
    md <- c(md,
      sprintf("## Subject %d (WT = %g)", subjects$id[i], subjects$wt[i]),
      "",
      "| decision | time (h) | trough (mg/L) | fired | amt (mg) | margin | raw (mg) |",
      "|---:|---:|---:|:---|---:|---:|---:|",
      vapply(seq_len(nrow(r)), function(j) sprintf(
        "| %d | %d | %s | %s | %s | %s | %s |",
        as.integer(r$decision[j]), as.integer(r$time[j]), fmt(r$trough[j]),
        r$fired[j], fmt(r$amt[j]),
        if (is.na(r$margin[j])) "" else fmt(r$margin[j]),
        if (is.na(r$raw[j])) "" else fmt(r$raw[j])), character(1)),
      "")
  }
  md <- c(md, sprintf("Smallest margin after t = 0: %s mg/L.", fmt(min_margin)))
  cat(sprintf("\nround_to min margin = %g\n", min_margin))
  writeLines(md, path)
  cat(sprintf("wrote %s\n", path))
}

write_plain(data.frame(id = c(1, 2), wt = c(50, 150)), "expected.md")
write_rounded(data.frame(id = c(1, 2), wt = c(40, 80)), "expected_round_to.md",
              lo = 250, step = 250)
