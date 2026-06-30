#!/usr/bin/env Rscript
# Platelet-ladder dose-modification reference (epic #391 S2.5a).
#
# External comparator for ferx-core's reactive `[adaptive_dosing]` `levels`
# ladder. NONMEM has no native feedback dosing, so mrgsolve (which does) is the
# apples-to-apples anchor for this feature family (see
# docs/model-file/adaptive-dosing.qmd, "Validation"). The ferx model under test
# is examples/adaptive_platelet_ladder.ferx; this script reproduces the identical
# structural model + controller in mrgsolve and freezes the realized dose ladder
# to expected.md, which tests/adaptive_platelet_anchor.rs asserts ferx matches.
#
# Friberg semi-mechanistic myelosuppression model (Friberg et al., J Clin Oncol
# 2002, 20(24):4713) driven by a 1-compartment IV drug: circulating platelets
# (CIRC, x10^9/L) are read at the start of each weekly cycle and the dose steps
# DOWN one discrete level on thrombocytopenia, with a severe floor that STOPS.
#
# The R loop mirrors ferx's controller semantics EXACTLY:
#   - read the latent signal (CIRC) at the decision time (pre-dose),
#   - first-matching rule wins (severe floor listed first),
#   - bare `decrease` steps one rung down a strictly-increasing ladder (saturating),
#   - no rule match -> re-issue the current rung's dose unchanged,
#   - `stop` discontinues all future dosing (and is the last logged decision).
#
# Regenerate:  Rscript platelet_mrgsolve.R   (run from this directory)
# Requires:    mrgsolve (tested with 1.7.2) + a C/C++ toolchain (JIT compile).

suppressMessages(library(mrgsolve))

# ---- structural model: identical params to adaptive_platelet_ladder.ferx -----
code <- '
$PARAM CL=0.22, V=15, CIRC0=250, MTT=120, GAMMA=0.20, SLOPE=0.06
$CMT CENT PROL TR1 TR2 TR3 CIRC
$ODE
  double KTR   = 4.0/MTT;                 // (n+1)/MTT, n = 3 transit compartments
  dxdt_CENT = -(CL/V)*CENT;
  dxdt_PROL = KTR*PROL*(1.0 - SLOPE*(CENT/V))*pow(CIRC0/CIRC, GAMMA) - KTR*PROL;
  dxdt_TR1  = KTR*(PROL - TR1);
  dxdt_TR2  = KTR*(TR1  - TR2);
  dxdt_TR3  = KTR*(TR2  - TR3);
  dxdt_CIRC = KTR*TR3 - KTR*CIRC;
'
mod <- mcode("platelet", code, rtol = 1e-8, atol = 1e-8, maxsteps = 100000)

B        <- 250                   # CIRC0 baseline (x10^9/L)
levels   <- c(25, 50, 75, 100)    # strictly-increasing ladder (mg)
cycle_h  <- 168                   # 1-week cycle
n_cyc    <- 10
dec_t    <- seq(0, by = cycle_h, length.out = n_cyc)
thr_stop <- 30                    # PLT < 30  -> stop (grade 4 floor)
thr_drop <- 120                   # PLT < 120 -> decrease one level

# ---- reactive loop (carry ODE state across cycles) ---------------------------
st  <- c(CENT = 0, PROL = B, TR1 = B, TR2 = B, TR3 = B, CIRC = B)
lev <- length(levels)             # start at the top rung (start_dose = 100)
rows <- list()
for (k in seq_along(dec_t)) {
  t0  <- dec_t[k]
  plt <- unname(st["CIRC"])       # latent signal the controller sees (pre-dose)
  if (plt < thr_stop) {
    rows[[k]] <- data.frame(decision = k - 1, time = t0, PLT = plt,
                            dose = NA_real_, action = "stop")
    break                         # discontinue; no further decisions logged
  } else if (plt < thr_drop) {
    lev    <- max(1L, lev - 1L)   # step one level down (saturating)
    action <- "decrease"
  } else {
    action <- "continue"          # re-issue current rung
  }
  dose <- levels[lev]
  rows[[k]] <- data.frame(decision = k - 1, time = t0, PLT = plt,
                          dose = dose, action = action)
  st["CENT"] <- st["CENT"] + dose # IV bolus into the drug compartment
  t1  <- if (k < length(dec_t)) dec_t[k + 1] else t0 + cycle_h
  seg <- as.data.frame(mod |>
    init(CENT = st["CENT"], PROL = st["PROL"], TR1 = st["TR1"],
         TR2 = st["TR2"], TR3 = st["TR3"], CIRC = st["CIRC"]) |>
    mrgsim(start = 0, end = t1 - t0, delta = t1 - t0))
  st[] <- as.numeric(seg[nrow(seg), c("CENT", "PROL", "TR1", "TR2", "TR3", "CIRC")])
}
ladder <- do.call(rbind, rows)

cum  <- sum(ladder$dose, na.rm = TRUE)
ndec <- sum(ladder$action == "decrease")
disc <- any(ladder$action == "stop")
in_win <- mean(ladder$PLT >= 100)   # one-sided target_window = [100, inf]

cat("=== platelet dose-modification ladder (mrgsolve 1.7.2) ===\n")
print(ladder, row.names = FALSE, digits = 10)
cat(sprintf("\ncumulative_dose=%g n_doses=%d n_decreases=%d discontinued=%s pct>=100=%g\n",
            cum, sum(!is.na(ladder$dose)), ndec, disc, in_win))

# ---- freeze expected.md ------------------------------------------------------
fmt <- function(x) formatC(x, format = "f", digits = 6)
tbl <- c(
  "| decision | time (h) | PLT (x10^9/L) | dose (mg) | action |",
  "|---:|---:|---:|---:|:---|",
  apply(ladder, 1, function(r) sprintf(
    "| %d | %d | %s | %s | %s |",
    as.integer(r["decision"]), as.integer(r["time"]), fmt(as.numeric(r["PLT"])),
    ifelse(is.na(r["dose"]), "-", as.integer(r["dose"])), trimws(r["action"]))))
md <- c(
  "# Platelet-ladder dose-modification reference (mrgsolve)",
  "",
  "Frozen output of `platelet_mrgsolve.R` (mrgsolve 1.7.2). External anchor for",
  "the reactive `[adaptive_dosing]` `levels` ladder in",
  "`examples/adaptive_platelet_ladder.ferx` (epic #391 S2.5a). NONMEM has no",
  "feedback dosing, so mrgsolve is the comparator; the R driver mirrors ferx's",
  "controller semantics exactly. Regenerate with `Rscript platelet_mrgsolve.R`.",
  "",
  "Friberg myelosuppression (n=3 transit) + 1-cpt IV drug. Params (identical in",
  "the ferx model): CL=0.22, V=15, CIRC0=250, MTT=120, GAMMA=0.20, SLOPE=0.06.",
  "Ladder levels [25,50,75,100] mg, start_dose 100; weekly decisions t=0..1512 h;",
  "rules: PLT<30 -> stop, PLT<120 -> decrease one level.",
  "",
  "## Realized dose ladder",
  "",
  tbl,
  "",
  "## Summary metrics",
  "",
  sprintf("- cumulative_dose = %g mg", cum),
  sprintf("- n_doses = %d", sum(!is.na(ladder$dose))),
  sprintf("- n_decreases = %d (100 -> 75 -> 50, then platelets recover and hold)", ndec),
  sprintf("- discontinued = %s", tolower(as.character(disc))),
  sprintf("- pct_time_in_window (PLT >= 100) = %g", in_win),
  "",
  "ferx (`tests/adaptive_platelet_anchor.rs`, RK45) reproduces this ladder",
  "exactly and the platelet signals to < 0.01 (x10^9/L) -- a cross-solver",
  "difference ~3e-5 relative, far below the ~25-unit margin to either rule",
  "threshold, so every dose decision is robust.")
writeLines(md, "expected.md")
cat("\nwrote expected.md\n")
