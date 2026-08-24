#!/usr/bin/env Rscript
#
# Plots for the VI cross-implementation comparison (VI_VALIDATION.md Anchor B).
#
# Reads both sides' saved output -- the ferx fit YAMLs and emvi-results.rds -- and draws the
# four figures the comparison is actually read through. Run AFTER run.sh; it fits nothing.
#
#   1. population-vs-agq.png  Tier 1. Signed deviation from the AGQ reference, per parameter,
#                             for all five estimator columns.
#   2. eta-means-vs-nuts.png  Tier 2a. Per-subject posterior means as a deviation from the
#                             per-subject NUTS reference (Anchor C).
#   3. eta-variance.png       Tier 2b. Per-subject posterior variance by subject, log scale,
#                             both tools against the AGQ Laplace reference.
#   4. eta-variance-ratio.png Each tool's variance divided by that reference.
#
# EVERY figure is read against a reference that neither implementation computes -- AGQ for the
# population parameters and the per-subject variances, NUTS for the per-subject means. That is
# the correction the first run of this harness needed, and figures 1-2 were the last to get it:
# they used to plot ferx AGAINST nlmixr2, which measures mutual agreement and cannot say which
# side is right. For a while both were wrong in the same direction and the parity plot looked
# healthy (VI_VALIDATION.md 4.11a). Mutual agreement also flattered the wrong thing: nlmixr2's
# FOCEI is the loosest of the five columns (Omega 6-7% off AGQ), so "the VI pair agrees better
# than the FOCEI pair" was mostly a statement about nlmixr2's inner loop, not about VI.
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
#   FERX_VI_FIGS=./figures Rscript tools/vi-emvi-comparison/plots.R
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

# NONMEM 7.5.1 FOCEI -- `tests/nonmem/warfarin_imp.ext`, TABLE NO. 1 (`#METH: First Order
# Conditional Estimation with Interaction`), the first step of the control stream the IMP anchor
# already committed. Same model, same data, same initial estimates. It is in this figure to
# CORROBORATE the reference rather than to compete with it: AGQ integrates the marginal and
# NONMEM approximates it, and if the two disagreed the reference would be the thing in doubt.
NM_FOCEI <- c(1.32695e-01, 7.73771e+00, 8.10796e-01, 2.85884e-02, 9.59179e-03, 3.35880e-01,
              sqrt(1.11621e-04))

agq_pop <- pull_pop(agq)
column <- function(label, tool, family, value) tibble(
  estimator = label, tool = tool, family = family, parameter = factor(PAR, levels = rev(PAR)),
  pct = 100 * (value - agq_pop) / abs(agq_pop)
)
dev <- bind_rows(
  column("NONMEM FOCEI", "NONMEM", "FOCEI", NM_FOCEI),
  column("ferx FOCEI", "ferx", "FOCEI", pull_pop(f_focei)),
  column("nlmixr2 FOCEI", "nlmixr2", "FOCEI", nm_vec(e$focei)),
  column("ferx VI", "ferx", "VI", pull_pop(f_vi)),
  column("nlmixr2 emvi", "nlmixr2", "VI", nm_vec(e$emvi))
) |> mutate(estimator = factor(estimator, c("NONMEM FOCEI", "ferx FOCEI", "nlmixr2 FOCEI",
                                            "ferx VI", "nlmixr2 emvi")))

# ---- 1. population parameters, against the AGQ reference --------------------------------
# One row per parameter, one point per estimator, zero = the reference. This replaced a
# ferx-vs-nlmixr2 dumbbell, which plotted mutual agreement -- a quantity that cannot say which
# side is right, and that flattered the VI pair because nlmixr2's FOCEI is the loose column.
#
# Colour is the tool and never swaps between panels; shape is the estimator family. NONMEM
# appears once, in ink, because it is corroboration rather than a fourth competitor.
NEAR <- 1.0  # % -- inside this, a column is "on the reference" and goes unlabelled
lab1 <- dev |> filter(abs(pct) > NEAR)

# Dodged, not overplotted: three of the five columns sit within 0.05% of the reference on most
# parameters, so at a shared y they occlude each other and "everything is on the line" reads as
# one point rather than as agreement between four estimators. The dodge keeps each column on its
# own row-slot, in a fixed order, so a reader can follow one estimator across parameters.
DODGE <- position_dodge(width = 0.78)

p1 <- ggplot(dev, aes(pct, parameter, group = estimator)) +
  annotate("rect", xmin = -NEAR, xmax = NEAR, ymin = -Inf, ymax = Inf, fill = RULE, alpha = 0.45) +
  geom_vline(xintercept = 0, colour = REF, linewidth = 0.5) +
  geom_point(aes(fill = tool, shape = family), size = 2.6, colour = "white", stroke = 0.6,
             position = DODGE) +
  geom_text(data = lab1, aes(label = sprintf("%+.1f%%", pct)), hjust = -0.3, size = 2.6,
            colour = INK2, position = DODGE) +
  scale_fill_manual(values = c(ferx = COL[["ferx"]], nlmixr2 = COL[["nlmixr2"]], NONMEM = INK)) +
  scale_shape_manual(values = c(FOCEI = 24, VI = 21)) +
  guides(
    fill = guide_legend(order = 1, override.aes = list(shape = 21, size = 3)),
    shape = guide_legend(order = 2, override.aes = list(fill = INK3, size = 3))
  ) +
  scale_x_continuous(breaks = seq(-2, 8, 2), labels = label_percent(scale = 1, accuracy = 1),
                     expand = expansion(c(0.06, 0.16))) +
  labs(
    title = "Tier 1 — every estimator against the AGQ reference",
    # Computed rather than written in: this subtitle went stale once already when the runs
    # behind it were corrected, and a figure that misreports its own numbers is worse than one
    # carrying no numbers at all.
    subtitle = sprintf(paste0(
      "Signed deviation from ferx AGQ (n_agq = 9), which integrates the marginal instead of ",
      "approximating it.\nShaded band is ±%.0f%%. NONMEM's FOCEI lands on the reference (worst ",
      "%.2f%%), so the reference is corroborated,\nnot assumed — and ferx FOCEI reproduces NONMEM ",
      "to %.0e on the OFV. The outliers are nlmixr2's FOCEI Ω\n(up to %+.1f%%) and both VI ",
      "implementations' σ, which miss in OPPOSITE directions (%+.1f%% / %+.1f%%)."),
      NEAR,
      max(abs(dev$pct[dev$estimator == "NONMEM FOCEI"])),
      1e-6,
      max(dev$pct[dev$estimator == "nlmixr2 FOCEI"]),
      dev$pct[dev$estimator == "ferx VI" & dev$parameter == "σ"],
      dev$pct[dev$estimator == "nlmixr2 emvi" & dev$parameter == "σ"]),
    x = "deviation from AGQ", y = NULL,
    caption = paste0(
      "warfarin, 10 subjects, 110 observations · 1-cpt oral, lognormal η on CL/V/KA, ",
      "proportional error\nNONMEM column: tests/nonmem/warfarin_imp.ext TABLE NO. 1 · ",
      "σ is compared as an SD, so NONMEM's $SIGMA variance is square-rooted first")
  ) +
  theme_ferx() +
  theme(panel.grid.major.y = element_line(colour = RULE, linewidth = 0.3),
        panel.grid.major.x = element_line(colour = RULE, linewidth = 0.3))

ggsave(file.path(figs, "population-vs-agq.png"), p1, width = 8.2, height = 5.2, dpi = 200)

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

# ---- 2. per-subject means, against the NUTS reference -----------------------------------
# This was a ferx-vs-emvi parity plot. Two approximations on an identity line cannot say which
# one is right -- and Anchor C provides the quantity that can: per-subject NUTS on
# p(eta | y_i, theta_hat), 4 chains x 20 000 draws, 0 divergences. So both tools are drawn as a
# DEVIATION from that, and the identity line becomes a zero line with a meaning.
#
# Absolute, not relative: several subjects have a near-zero mean, where a percentage explodes on
# a difference of 1e-3 and reports a disagreement that is not there (id 6's eta_CL is -0.008, so
# 22% relative is 1.9e-03 absolute).
anchor_c <- file.path(res, "anchor-c.json")
if (!file.exists(anchor_c)) {
  message("skipping eta-means-vs-nuts.png: ", anchor_c, " not found\n",
          "  run tools/vi-nuts-anchor/run.sh to produce the per-subject NUTS reference")
} else {
  ac <- jsonlite::fromJSON(anchor_c)
  stopifnot("the NUTS reference must cover the same subjects as the fits" =
              nrow(ac$nuts_mean) == length(unique(q$id)))
  nuts <- bind_rows(lapply(seq_along(ac$ids), function(i) tibble(
    id = as.integer(ac$ids[i]), eta = factor(ETA, ETA), nuts = ac$nuts_mean[i, ])))

  dq <- q |> left_join(nuts, by = c("id", "eta")) |>
    transmute(id, eta, ferx = mean_ferx - nuts, nlmixr2 = mean_emvi - nuts) |>
    pivot_longer(c(ferx, nlmixr2), names_to = "tool", values_to = "d")

  p2 <- ggplot(dq, aes(factor(id), d)) +
    geom_hline(yintercept = 0, colour = REF, linewidth = 0.5) +
    geom_point(aes(fill = tool), shape = 21, size = 2.6, colour = "white", stroke = 0.6) +
    facet_wrap(~eta, nrow = 1) +
    scale_fill_manual(values = COL) +
    scale_y_continuous(labels = label_number(scale = 1e3, accuracy = 0.5)) +
    labs(
      title = "Tier 2a — per-subject posterior means, against per-subject NUTS",
      subtitle = sprintf(paste0(
        "Zero is the exact posterior mean, from NUTS at a fixed population estimate (Anchor C). ",
        "Both tools sit on it:\nferx within %.1e and emvi within %.1e, against an η range of ",
        "%.2f–%.2f across subjects. So the means are\nnot where the two implementations differ — ",
        "the variances are (figures 3–4), and σ is (figure 1)."),
        max(abs(dq$d[dq$tool == "ferx"])), max(abs(dq$d[dq$tool == "nlmixr2"])),
        min(apply(ac$nuts_mean, 2, function(z) diff(range(z)))),
        max(apply(ac$nuts_mean, 2, function(z) diff(range(z))))),
      x = "subject", y = "mean − NUTS mean  (×10⁻³)",
      caption = paste0(
        "absolute, not relative: several subjects have a near-zero mean, where a percentage ",
        "misreports a 10⁻³ gap\nemvi's q comes from its own fit (σ ~2% from AGQ's) while NUTS ",
        "and ferx's q both sit at the AGQ estimate,\nso part of the orange spread is that, not ",
        "the inference · ferx VI with the population parameters FIXed lands within 9.5e-05")
    ) + theme_ferx()

  ggsave(file.path(figs, "eta-means-vs-nuts.png"), p2, width = 8.4, height = 4.0, dpi = 200)
}

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
    caption = paste0(
      "diagonal only · the off-diagonals are compared separately, against NUTS rather than AGQ: ",
      "Rscript tools/vi-emvi-comparison/tier2-offdiag.R\nnlmixr2's packed Cholesky is row-major, ",
      "read from nlmixr2est's source (src/inner.cpp:15107 / :15293) — no longer an open question ",
      "(VI_VALIDATION.md 4.7)")
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
for (f in c("population-vs-agq.png", "eta-means-vs-nuts.png", "eta-variance.png", "eta-variance-ratio.png")) {
  fp <- file.path(figs, f)
  cat(sprintf("  %-26s %s\n", f,
              if (file.exists(fp)) sprintf("ok  %5.0f KB", file.size(fp) / 1024) else "MISSING"))
}
