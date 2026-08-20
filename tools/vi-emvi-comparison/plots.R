#!/usr/bin/env Rscript
#
# Plots for the VI cross-implementation comparison (VI_VALIDATION.md Anchor B).
#
# Reads both sides' saved output -- the ferx fit YAMLs and emvi-results.rds -- and draws the
# four figures the comparison is actually read through. Run AFTER run.sh; it fits nothing.
#
#   1. population-parity.png  Tier 1. |relative difference| per parameter, for the VI pair AND
#                             for the FOCEI pair. The FOCEI pair is the point: it is the same
#                             two codebases on an established method, so it calibrates what
#                             "agreement" means here before the VI numbers are read.
#   2. eta-means-parity.png   Tier 2. Per-subject variational means, ferx vs emvi, against the
#                             identity line. The pharmacometric parity plot.
#   3. eta-variance.png       Tier 2. Per-subject posterior variance by subject, log scale,
#                             both tools against the AGQ Laplace reference.
#   4. eta-variance-ratio.png Each tool's variance divided by that reference.
#
# Figures 3-4 are read against AGQ rather than against each other, which is the correction the
# first run of this harness needed: two approximations agreeing says nothing about which is
# right, and for a while both were wrong in the same direction (VI_VALIDATION.md 4.11).
#
# Usage:  Rscript tools/vi-emvi-comparison/plots.R
#
# Two directories, deliberately separate:
#   FERX_VI_OUT   where the fits are READ from  (default $FERX_VI_STATE/results)
#   FERX_VI_FIGS  where the figures are WRITTEN (default: the same place)
#
# They are split because the figures are usually wanted somewhere a person actually looks --
# a project folder, a report directory -- while the inputs stay in the harness state dir. Point
# only FERX_VI_FIGS at the destination; pointing FERX_VI_OUT there instead would make the script
# look for emvi-results.rds in the figure folder and fail.
#
#   FERX_VI_FIGS=~/Downloads/dcm_comparison Rscript tools/vi-emvi-comparison/plots.R
suppressMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(scales); library(patchwork)
  library(numDeriv)
})

state <- Sys.getenv("FERX_VI_STATE", file.path(Sys.getenv("HOME"), ".local/share/ferx-vi-validation"))
res <- Sys.getenv("FERX_VI_OUT", file.path(state, "results"))
stopifnot("results directory not found -- run run.sh first" = dir.exists(res))

# Fail early and by name if an input is missing, rather than at the first yaml::read_yaml with a
# path buried in the message. warfarin_cmp is the FOCEI baseline the whole Tier 1 figure rests on.
need <- c("warfarin_cmp-fit.yaml", "vi_closed_form-fit.yaml", "emvi-results.rds")
missing <- need[!file.exists(file.path(res, need))]
if (length(missing)) {
  stop(sprintf("missing input(s) in %s: %s\n  run tools/vi-emvi-comparison/run.sh first",
               res, paste(missing, collapse = ", ")), call. = FALSE)
}

figs <- Sys.getenv("FERX_VI_FIGS", res)
dir.create(figs, showWarnings = FALSE, recursive = TRUE)

# Palette: ferx / nlmixr2 are fixed to an entity each and never swap between panels. Validated
# for CVD separation and contrast against both a light and a dark surface.
COL <- c(ferx = "#2a78d6", nlmixr2 = "#eb6834")
REF <- "#8b9299"
INK <- "#14181c"; INK2 <- "#4a5259"; INK3 <- "#767f86"; RULE <- "#e2e5e0"

theme_ferx <- function(base = 11) {
  theme_minimal(base_size = base, base_family = "") +
    theme(
      plot.title = element_text(face = "bold", size = rel(1.05), colour = INK, hjust = 0),
      plot.subtitle = element_text(size = rel(0.88), colour = INK2, hjust = 0, lineheight = 1.25,
                                   margin = margin(t = 3, b = 11)),
      plot.caption = element_text(size = rel(0.74), colour = INK3, hjust = 0,
                                  margin = margin(t = 11)),
      plot.title.position = "plot", plot.caption.position = "plot",
      axis.title = element_text(size = rel(0.82), colour = INK2),
      axis.text = element_text(size = rel(0.78), colour = INK3),
      panel.grid.major = element_line(colour = RULE, linewidth = 0.3),
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold", size = rel(0.85), colour = INK, hjust = 0),
      legend.position = "top", legend.justification = "left",
      legend.title = element_blank(), legend.text = element_text(size = rel(0.82), colour = INK2),
      legend.key.size = unit(9, "pt"), legend.margin = margin(b = 2),
      plot.margin = margin(14, 16, 12, 14), plot.background = element_rect(fill = "white", colour = NA)
    )
}

# ---- load -------------------------------------------------------------------------------
fx <- function(f) yaml::read_yaml(file.path(res, f))
pull_pop <- function(y) c(
  vapply(y$theta, function(t) t$estimate, numeric(1)),
  vapply(y$omega, function(o) o$variance, numeric(1)),
  sigma = y$sigma[[1]]$estimate
)

# ---- the AGQ reference ------------------------------------------------------------------
# H^-1 at the AGQ estimate: the per-subject posterior covariance a converged Gaussian q should
# be reporting. Computed here, in neither codebase, from the closed form both of them evaluate.
laplace_reference <- function(agq) {
  dat <- read.csv(Sys.getenv("FERX_DATA", "data/warfarin.csv"), na.strings = ".")
  dat$DV <- as.numeric(dat$DV); dat$AMT <- as.numeric(dat$AMT)
  ids <- sort(unique(dat$ID))
  o0 <- subset(dat, EVID == 0 & !is.na(DV))
  obs <- split(o0, o0$ID)
  dose <- sapply(split(dat, dat$ID), function(d) sum(d$AMT[d$EVID == 1], na.rm = TRUE))
  th <- vapply(agq$theta, function(t) t$estimate, numeric(1))
  om <- vapply(agq$omega, function(o) o$variance, numeric(1))
  sg <- agq$sigma[[1]]$estimate
  # 1-cpt oral, single bolus into the depot, F = 1 -- the closed form ferx's one_cpt_oral and
  # nlmixr2's linCmt() both evaluate.
  pred <- function(t, cl, v, ka, D) { ke <- cl / v; D * ka / (v * (ka - ke)) * (exp(-ke * t) - exp(-ka * t)) }
  t(sapply(ids, function(id) {
    o <- obs[[as.character(id)]]
    fn <- function(e) {
      f <- pred(o$TIME, th[1] * exp(e[1]), th[2] * exp(e[2]), th[3] * exp(e[3]),
                dose[[as.character(id)]])
      sd <- sg * f
      0.5 * sum(log(sd^2) + (o$DV - f)^2 / sd^2) + 0.5 * sum(e^2 / om)
    }
    m <- optim(c(0, 0, 0), fn, method = "BFGS", control = list(reltol = 1e-12))$par
    diag(solve(numDeriv::hessian(fn, m)))
  }))
}
agq <- fx("agq_ref-fit.yaml")
Href <- laplace_reference(agq)

f_focei <- fx("warfarin_cmp-fit.yaml")
f_vi <- fx("vi_closed_form-fit.yaml")
e <- readRDS(file.path(res, "emvi-results.rds"))

PAR <- c("TVCL", "TVV", "TVKA", "ω²CL", "ω²V", "ω²KA", "σ")
nm_vec <- function(x) c(x$est[c("tvcl", "tvv", "tvka")], diag_or(x$omega), x$est[["prop.err"]])
diag_or <- function(om) if (is.matrix(om)) diag(om) else om

pop <- tibble(
  parameter = factor(PAR, levels = rev(PAR)),
  # Same two codebases, established method: the yardstick for what agreement means here.
  focei = abs(nm_vec(e$focei) - pull_pop(f_focei)) / abs(pull_pop(f_focei)) * 100,
  vi    = abs(nm_vec(e$emvi) - pull_pop(f_vi)) / abs(pull_pop(f_vi)) * 100
)

# ---- 1. population parameters -----------------------------------------------------------
# Dumbbell rather than grouped bars: the reader's question is "did the VI pair land closer or
# further than the FOCEI pair on this parameter", which is a shift, and a shift reads as a
# segment. Only the extremes are labelled -- a number on all 14 points would go unread.
lp <- pop |> pivot_longer(c(focei, vi), names_to = "pair", values_to = "pct") |>
  mutate(pair = factor(pair, c("focei", "vi"),
                       c("FOCEI vs FOCEI  (cross-tool baseline)", "ferx VI vs emvi")))
lab <- lp |> group_by(pair) |> slice_max(pct, n = 1) |> ungroup()

p1 <- ggplot(pop, aes(y = parameter)) +
  geom_segment(aes(x = focei, xend = vi, yend = parameter), colour = REF, linewidth = 0.4) +
  geom_point(data = lp, aes(x = pct, fill = pair), shape = 21, size = 3.1, colour = "white",
             stroke = 0.7) +
  geom_text(data = lab, aes(x = pct, label = sprintf("%.2f%%", pct)),
            hjust = -0.28, size = 2.9, colour = INK2, family = "") +
  scale_fill_manual(values = c(REF, COL[["ferx"]])) +
  scale_x_continuous(labels = label_percent(scale = 1, accuracy = 1), expand = expansion(c(0.02, 0.14))) +
  labs(
    title = "Tier 1 — population parameters agree, and the VI pair beats the FOCEI baseline",
    subtitle = paste0(
      "Absolute relative difference between ferx and nlmixr2, per parameter. The grey series is the ",
      "same two\ncodebases on FOCEI — an established method — so it calibrates the tolerance these ",
      "tools show each other\nbefore the VI numbers are read. The VI pair is tighter on every ",
      "parameter but σ: ~40× on the fixed\neffects, ~10× on Ω. Both arms converged (ferx carries the ",
      "closed-form σ M-step; emvi runs 20k iters)."),
    x = "|relative difference|", y = NULL,
    caption = paste0(
      "warfarin, 10 subjects, 110 observations · 1-cpt oral, lognormal η on CL/V/KA, proportional ",
      "error\nσ is the one row where the gap is one-sided rather than mutual: against AGQ's ",
      "0.010565, ferx sits +6.3% and emvi −1.9%")
  ) +
  theme_ferx() + theme(panel.grid.major.y = element_blank())

ggsave(file.path(figs, "population-parity.png"), p1, width = 7.6, height = 4.5, dpi = 200)

# ---- per-subject q ----------------------------------------------------------------------
ETA <- c("η CL", "η V", "η KA")
ferx_q <- bind_rows(lapply(f_vi$vi$eta_posterior, function(s) tibble(
  id = as.integer(s$id), eta = ETA,
  mean = unlist(s$mean),
  var = vapply(seq_along(ETA), function(j) s$cov[[j]][[j]], numeric(1))
)))
emvi_q <- bind_rows(lapply(seq_len(nrow(e$q$mu)), function(i) tibble(
  id = i, eta = ETA, mean = e$q$mu[i, ], var = diag(e$q$cov[[i]])
)))
q <- full_join(ferx_q, emvi_q, by = c("id", "eta"), suffix = c("_ferx", "_emvi")) |>
  mutate(eta = factor(eta, ETA))

# ---- 2. variational means: the parity plot ----------------------------------------------
# Free scales per facet on purpose: eta_KA spans 2.0 and eta_V spans 0.3, and a shared scale
# would compress eta_V into a dot. The identity line is what carries the comparison, and it is
# the same line in every panel regardless of range.
# Annotated top-left rather than beside the point: a label anchored to an extreme point at a
# panel edge gets clipped by the panel, and nudging it inward detaches it from its dot.
worst <- q |> mutate(d = abs(mean_emvi - mean_ferx)) |> group_by(eta) |>
  slice_max(d, n = 1) |> ungroup() |> mutate(lab = sprintf("worst: id %d,  Δ %.1e", id, d))

p2a <- ggplot(q, aes(mean_ferx, mean_emvi)) +
  geom_abline(slope = 1, intercept = 0, colour = REF, linewidth = 0.4) +
  geom_point(fill = COL[["ferx"]], shape = 21, size = 2.5, colour = "white", stroke = 0.6) +
  geom_text(data = worst, aes(x = -Inf, y = Inf, label = lab), inherit.aes = FALSE,
            hjust = -0.07, vjust = 1.5, size = 2.5, colour = INK2) +
  facet_wrap(~eta, scales = "free", nrow = 1) +
  scale_x_continuous(n.breaks = 4) +
  scale_y_continuous(n.breaks = 4, expand = expansion(c(0.05, 0.16))) +
  labs(
    title = "Tier 2a — per-subject variational means: parity",
    # Computed, not written in: this subtitle went stale once already when the underlying runs
    # were corrected, and a figure that misreports its own numbers is worse than one with none.
    subtitle = sprintf(paste0(
      "Top: each subject's posterior mean, ferx against emvi, on the identity line. Bottom: the same ",
      "comparison as\na signed difference, which is the only way to see a residual this small — every ",
      "point above sits on the line.\nLargest disagreement anywhere is %.1e, under %.1f%% of the η ",
      "range the subjects span."),
      max(abs(q$mean_emvi - q$mean_ferx)),
      100 * max(abs(q$mean_emvi - q$mean_ferx)) /
        max(q |> group_by(eta) |> summarise(r = diff(range(mean_ferx)), .groups = "drop") |> pull(r))),
    x = NULL, y = "emvi mean"
  ) + theme_ferx() + theme(plot.margin = margin(14, 16, 2, 14))

# Absolute difference, not relative: several subjects have a near-zero mean, where a percentage
# explodes on a difference of 10⁻³ and reports a disagreement that is not there (id 6's η CL is
# -0.008, so 22% relative is 1.9e-03 absolute).
p2b <- ggplot(q, aes(factor(id), mean_emvi - mean_ferx)) +
  geom_hline(yintercept = 0, colour = REF, linewidth = 0.4) +
  geom_point(fill = COL[["ferx"]], shape = 21, size = 2.5, colour = "white", stroke = 0.6) +
  facet_wrap(~eta, nrow = 1) +
  scale_y_continuous(labels = label_number(scale = 1e3, accuracy = 1)) +
  labs(x = "subject", y = "difference (×10⁻³)",
       caption = "identity line, not a fit · differences are absolute, not relative: several subjects have a near-zero mean, where a percentage misreports a 10⁻³ gap") +
  theme_ferx() + theme(strip.text = element_blank(), plot.margin = margin(2, 16, 12, 14))

p2 <- patchwork::wrap_plots(p2a, p2b, ncol = 1, heights = c(1, 0.72))
ggsave(file.path(figs, "eta-means-parity.png"), p2, width = 8.2, height = 6.0, dpi = 200)

# ---- 3. posterior variances against the reference ---------------------------------------
# Subject on x and log variance on y, rather than a parity plot: the story is the SHAPE of each
# tool's spread across subjects, which a parity plot hides behind the identity line. The grey
# reference is AGQ's H^-1 -- recessive because it is the truth line, not a third competitor.
lv <- q |> select(id, eta, ferx = var_ferx, nlmixr2 = var_emvi) |>
  pivot_longer(c(ferx, nlmixr2), names_to = "tool", values_to = "var")
ref <- tibble(id = rep(seq_len(nrow(Href)), 3), eta = factor(rep(ETA, each = nrow(Href)), ETA),
              var = as.vector(Href))
spread <- q |> group_by(eta) |>
  summarise(f = max(var_ferx) / min(var_ferx), e = max(var_emvi) / min(var_emvi), .groups = "drop") |>
  mutate(lab = sprintf("across-subject spread   ferx %.1f×    emvi %.1f×", f, e))

p3 <- ggplot(lv, aes(factor(id), var)) +
  geom_point(data = ref, aes(shape = "AGQ reference (H⁻¹)"), colour = REF, size = 2.6) +
  geom_point(aes(fill = tool), shape = 21, size = 2.6, colour = "white", stroke = 0.6) +
  geom_text(data = spread, aes(x = 0.6, y = Inf, label = lab), inherit.aes = FALSE,
            hjust = 0, vjust = 1.6, size = 2.55, colour = INK2) +
  facet_wrap(~eta, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = COL) +
  scale_shape_manual(values = 95) +
  scale_y_log10(labels = label_scientific(digits = 2), expand = expansion(c(0.06, 0.22))) +
  labs(
    title = "Tier 2b — per-subject posterior variances, against a near-exact reference",
    subtitle = paste0(
      "Diagonal of each subject's variational covariance, log scale, with AGQ's Laplace ",
      "posterior H⁻¹ as the\nreference. Both implementations track it: medians 1.12 / 1.13 / 1.13 ",
      "(ferx) and 1.05 / 0.97 / 1.18 (emvi).\nferx's uniform ~12% excess is its σ still sitting ",
      "6.3% high — posterior width scales with σ²."),
    x = "subject", y = "posterior variance (log scale)",
    caption = "diagonal only · off-diagonals are not compared: nlmixr2's packed-Cholesky row- vs column-major order is unresolved (VI_VALIDATION.md 4.7)"
  ) + theme_ferx() + theme(legend.box = "horizontal")
ggsave(file.path(figs, "eta-variance.png"), p3, width = 8.4, height = 3.9, dpi = 200)

# ---- 4. the same thing as a ratio against the reference ---------------------------------
# The ratio is what separates a systematic offset from scatter, and taking it against AGQ rather
# than against the other tool is what lets it say *which* tool is off.
rat <- q |>
  transmute(id, eta,
            ferx = var_ferx / as.vector(Href)[match(paste(id, eta), paste(rep(seq_len(nrow(Href)), 3),
                     rep(ETA, each = nrow(Href))))],
            nlmixr2 = var_emvi / as.vector(Href)[match(paste(id, eta), paste(rep(seq_len(nrow(Href)), 3),
                     rep(ETA, each = nrow(Href))))]) |>
  pivot_longer(c(ferx, nlmixr2), names_to = "tool", values_to = "r")

p4 <- ggplot(rat, aes(factor(id), r, fill = tool)) +
  geom_hline(yintercept = 1, colour = REF, linewidth = 0.4) +
  geom_point(shape = 21, size = 2.6, colour = "white", stroke = 0.6) +
  facet_wrap(~eta, nrow = 1) +
  scale_fill_manual(values = COL) +
  scale_y_log10(breaks = c(0.5, 0.75, 1, 1.5, 2), labels = c("0.5×", "0.75×", "1×", "1.5×", "2×")) +
  labs(
    title = "Tier 2b, as a ratio — how far off each tool is, per subject",
    subtitle = paste0(
      "Each tool's posterior variance divided by the AGQ reference. On the line means correct. ",
      "The two miss it\ndifferently: ferx is a tight band ~12% high (a uniform σ² effect), emvi is ",
      "centred on it but scatters\n(0.65–1.51× against ferx's 1.11–1.45×). The earlier reading — a ",
      "24× spread and a 30% systematic\ndeficit — was an under-converged run on both sides."),
    x = "subject", y = "variance ÷ AGQ reference",
    caption = "log scale · a point below the line means the tool reports the tighter posterior"
  ) + theme_ferx()
ggsave(file.path(figs, "eta-variance-ratio.png"), p4, width = 8.4, height = 3.6, dpi = 200)

cat("read  from", res, "\n")
cat("wrote to  ", figs, "\n")
for (f in c("population-parity.png", "eta-means-parity.png", "eta-variance.png", "eta-variance-ratio.png")) {
  fp <- file.path(figs, f)
  cat(sprintf("  %-26s %s\n", f,
              if (file.exists(fp)) sprintf("ok  %5.0f KB", file.size(fp) / 1024) else "MISSING"))
}
