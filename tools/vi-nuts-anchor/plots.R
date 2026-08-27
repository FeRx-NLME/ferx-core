#!/usr/bin/env Rscript
#
# Figures for Anchor C (VI_VALIDATION.md §5.1) — the variational posterior against a per-subject
# NUTS reference. Run after run.sh; reads the JSON it writes and fits nothing.
#
#   1. posterior-overlay.png    The direct comparison: each subject's NUTS marginal with the
#                               variational Gaussian drawn over it. Summaries can only show two
#                               covariances agreeing; this shows the *shape* agreeing.
#   2. anchor-c-variance.png    VI variance ÷ NUTS variance, per subject, both data regimes.
#                               This is the published understatement claim, measured.
#   3. anchor-c-why.png         The three ratios side by side. `Laplace ÷ NUTS` is the
#                               interpretive key: at 1.00 the true posterior is Gaussian, so a
#                               small number in the first panel means the dataset cannot test the
#                               claim — not that VI is unusually good.
#
# Usage:  Rscript tools/vi-nuts-anchor/plots.R
# Output: $FERX_VI_FIGS (default: alongside the inputs)
suppressMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(scales); library(jsonlite)
})

state <- Sys.getenv("FERX_VI_STATE", file.path(Sys.getenv("HOME"), ".local/share/ferx-vi-validation"))
res <- Sys.getenv("FERX_VI_OUT", file.path(state, "results"))
figs <- Sys.getenv("FERX_VI_FIGS", res)
dir.create(figs, showWarnings = FALSE, recursive = TRUE)

# ferx blue for the variational posterior; the reference is recessive grey, because NUTS here is
# the truth line rather than a competing series. Same convention the emvi figures use for AGQ.
FERX <- "#2a78d6"
REF <- "#8b9299"
INK <- "#14181c"; INK2 <- "#4a5259"; INK3 <- "#767f86"; RULE <- "#e2e5e0"
ETA <- c("η CL", "η V", "η KA")

theme_ferx <- function(base = 11) {
  theme_minimal(base_size = base) +
    theme(
      plot.title = element_text(face = "bold", size = rel(1.05), colour = INK, hjust = 0),
      plot.subtitle = element_text(size = rel(0.88), colour = INK2, hjust = 0, lineheight = 1.25,
                                   margin = margin(t = 3, b = 11)),
      plot.caption = element_text(size = rel(0.74), colour = INK3, hjust = 0, margin = margin(t = 11)),
      plot.title.position = "plot", plot.caption.position = "plot",
      axis.title = element_text(size = rel(0.82), colour = INK2),
      axis.text = element_text(size = rel(0.78), colour = INK3),
      panel.grid.major = element_line(colour = RULE, linewidth = 0.3),
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold", size = rel(0.82), colour = INK, hjust = 0),
      legend.position = "top", legend.justification = "left",
      legend.title = element_blank(), legend.text = element_text(size = rel(0.82), colour = INK2),
      legend.key.size = unit(9, "pt"), legend.margin = margin(b = 2),
      plot.margin = margin(14, 16, 12, 14), plot.background = element_rect(fill = "white", colour = NA)
    )
}

load_run <- function(file, label) {
  stopifnot("Anchor C output not found -- run run.sh first" = file.exists(file))
  d <- fromJSON(file, simplifyVector = TRUE)
  n <- length(d$ids)
  # Ratios of the covariance *diagonals*: the off-diagonals are compared in the overlay instead,
  # where a wrong correlation shows up as a tilted contour rather than a number.
  diag_of <- function(a) t(sapply(seq_len(n), function(i) diag(a[i, , ])))
  list(
    label = label, ids = d$ids, draws = d$draws_thinned,
    vi_mean = d$vi_mean, vi_cov = d$vi_cov,
    ratios = bind_rows(lapply(
      list("VI ÷ NUTS" = diag_of(d$vi_cov) / diag_of(d$nuts_cov),
           "Laplace ÷ NUTS" = diag_of(d$laplace_cov) / diag_of(d$nuts_cov),
           "VI ÷ Laplace" = diag_of(d$vi_cov) / diag_of(d$laplace_cov)),
      function(m) {
        colnames(m) <- ETA
        as_tibble(m) |> mutate(id = seq_len(n))
      }
    ), .id = "quantity") |>
      pivot_longer(all_of(ETA), names_to = "eta", values_to = "ratio") |>
      mutate(eta = factor(eta, ETA), regime = label)
  )
}

# The regime strip labels the ARM, and its default is derived rather than hardcoded: the 1% arm
# lives in `results`, the realistic one in `results-10pct`. FERX_ARM_LABEL overrides.
arm_label <- Sys.getenv("FERX_ARM_LABEL", unset = {
  b <- basename(res)
  if (b == "results") "warfarin (~1% residual)" else sub("^results-", "", b)
})
full <- load_run(file.path(res, "anchor-c.json"), arm_label)
# The thinned variant is optional: it exists for the ~1% arm, where the posterior is Gaussian in
# both the data-rich and data-poor limits and sparsity was the only lever available. An arm
# without it simply gets one regime.
sparse_file <- file.path(res, "anchor-c-sparse.json")
sparse <- if (file.exists(sparse_file)) {
  load_run(sparse_file, "2 obs/subject")
} else {
  message("no anchor-c-sparse.json in ", res, " -- drawing the single regime")
  list(ratios = NULL)
}

# Everything the subtitles claim is computed. They used to be typed in, and they carried the ~1%
# arm's findings: "variances agree to 0.2%", "the true posterior IS Gaussian", "the published
# 20-25% understatement does not appear". On the 10% arm all three are false -- VI understates
# 3-4%, Laplace/NUTS is 0.966 rather than 1.000, so the posterior is NOT Gaussian to 0.1%, and an
# understatement does appear. A figure that narrates the wrong dataset is worse than a bare one.
med_of <- function(q) {
  full$ratios |> filter(quantity == q) |> group_by(eta) |>
    summarise(m = median(ratio), .groups = "drop")
}
fmt_med <- function(q) paste(sprintf("%.3f", med_of(q)$m), collapse = " / ")
worst_under <- function(q) 100 * (1 - min(med_of(q)$m))
gaussian_gap <- worst_under("Laplace ÷ NUTS")   # how far the true posterior is from Gaussian
vi_gap <- worst_under("VI ÷ NUTS")              # how far q is from the truth
nuts_mean <- fromJSON(file.path(res, "anchor-c.json"), simplifyVector = TRUE)$nuts_mean
mean_err <- max(abs(full$vi_mean - nuts_mean))

# ---- 1. the overlay ---------------------------------------------------------------------
# Four subjects rather than ten: the point is that the curves sit on the histograms, and ten
# panels of that reads as wallpaper. Subjects are taken across the range of posterior widths so
# the sample is not the flattering end of it.
w <- sapply(seq_along(full$ids), function(i) full$vi_cov[i, 3, 3])
pick <- order(w)[round(seq(1, length(w), length.out = 4))]
dr <- bind_rows(lapply(pick, function(i) bind_rows(lapply(1:3, function(j) tibble(
  id = full$ids[i], eta = ETA[j], draw = full$draws[, i, j]
)))))
gauss <- bind_rows(lapply(pick, function(i) bind_rows(lapply(1:3, function(j) {
  m <- full$vi_mean[i, j]; s <- sqrt(full$vi_cov[i, j, j])
  x <- seq(m - 4 * s, m + 4 * s, length.out = 200)
  tibble(id = full$ids[i], eta = ETA[j], x = x, d = dnorm(x, m, s))
}))))
# facet_wrap with fully independent scales, not facet_grid: a grid shares the x range down each
# column, so a subject whose η sits at 0.29 and one at −0.02 get the same axis and both densities
# collapse to spikes. The claim being made is about *shape*, so each panel gets its own range.
lvl <- unlist(lapply(full$ids[pick], function(i) paste0("subject ", i, "  ·  ", ETA)))
panel_of <- function(d) factor(paste0("subject ", d$id, "  ·  ", d$eta), levels = lvl)
dr$panel <- panel_of(dr); gauss$panel <- panel_of(gauss)

p1 <- ggplot(dr, aes(draw)) +
  geom_histogram(aes(y = after_stat(density)), bins = 55, fill = REF, colour = NA, alpha = 0.55) +
  geom_line(data = gauss, aes(x, d), colour = FERX, linewidth = 0.8) +
  facet_wrap(~panel, scales = "free", ncol = 3) +
  labs(
    title = "Anchor C — the variational posterior is the posterior",
    subtitle = sprintf(paste0(
      "Grey: the NUTS marginal for one subject's η, from 4 chains × 20 000 draws with 0 ",
      "divergences. Blue: the\nvariational Gaussian ferx reports, at the same fixed (θ, Ω, σ). ",
      "Four subjects spanning the range of posterior\nwidths. Median variance ratio %s; means ",
      "agree to %.1e."),
      fmt_med("VI ÷ NUTS"), mean_err),
    x = "η", y = "density",
    caption = sprintf(paste0(
      "population parameters FIXed at the AGQ estimate, so the comparison is about q alone · ",
      "results read from %s"), res)
  ) +
  theme_ferx() +
  theme(
    axis.text.y = element_blank(), panel.grid.major.y = element_blank(),
    strip.text = element_text(face = "plain", size = rel(0.74), colour = INK2, hjust = 0)
  )
ggsave(file.path(figs, "posterior-overlay.png"), p1, width = 8.4, height = 6.4, dpi = 200)

# ---- 2. the published claim, measured ---------------------------------------------------
vr <- bind_rows(full$ratios, sparse$ratios) |>
  filter(quantity == "VI ÷ NUTS") |>
  mutate(regime = factor(regime, unique(regime)))
med <- vr |> group_by(regime, eta) |> summarise(m = median(ratio), .groups = "drop")
p2 <- ggplot(vr, aes(factor(id), ratio)) +
  geom_hline(yintercept = 1, colour = REF, linewidth = 0.4) +
  geom_point(fill = FERX, shape = 21, size = 2.5, colour = "white", stroke = 0.6) +
  geom_text(data = med, aes(x = 0.6, y = Inf, label = sprintf("median %.3f", m)),
            inherit.aes = FALSE, hjust = 0, vjust = 1.6, size = 2.6, colour = INK2) +
  facet_grid(regime ~ eta) +
  scale_y_continuous(labels = label_number(accuracy = 0.01)) +
  labs(
    title = sprintf(
      "The understatement, measured: %.1f%% here against the published 20–25%%",
      vi_gap),
    subtitle = sprintf(paste0(
      "Variational variance ÷ the exact posterior's, per subject. On the line means no ",
      "understatement.\nMedian ratio %s, i.e. up to %.1f%% understated. The published 20–25%% is ",
      "from deep compartment models,\nwhose posterior geometry a 1-cpt model cannot produce — so ",
      "read this as the effect's size in THIS regime,\nnot as a refutation. The next figure ",
      "separates the family from the optimizer."),
      fmt_med("VI ÷ NUTS"), vi_gap),
    x = "subject", y = "VI variance ÷ NUTS variance",
    caption = "diagonals only · the off-diagonals are compared in the overlay, where a wrong correlation shows as a tilted density rather than a number"
  ) + theme_ferx()
ggsave(file.path(figs, "anchor-c-variance.png"), p2, width = 8.4, height = 4.6, dpi = 200)

# ---- 3. why it is small ------------------------------------------------------------------
# The middle panel is the one that stops the first figure being over-read: a small
# understatement against a Gaussian truth is not evidence of a good approximation, it is
# evidence the dataset cannot test the approximation.
ord <- c("VI ÷ NUTS", "Laplace ÷ NUTS", "VI ÷ Laplace")
# Regime by facet rather than by hue. Orange is nlmixr2 across the rest of this figure set, and
# these figures land in the same directory, so spending it on "data regime" here would make one
# colour mean two things to anyone reading them together. Faceting keeps a single blue series.
wr <- bind_rows(full$ratios, sparse$ratios) |>
  mutate(quantity = factor(quantity, ord),
         regime = factor(regime, unique(regime)))
p3 <- ggplot(wr, aes(eta, ratio)) +
  geom_hline(yintercept = 1, colour = REF, linewidth = 0.4) +
  geom_point(position = position_jitter(width = 0.18, height = 0, seed = 1),
             colour = FERX, size = 1.9, stroke = 0) +
  facet_grid(regime ~ quantity) +
  labs(
    title = "Where the understatement comes from: the family, or the optimizer",
    subtitle = sprintf(paste0(
      "Middle panel is the key: Laplace ÷ NUTS is %s, so the exact posterior departs from ",
      "Gaussian by %.1f%%.\nRight panel, VI ÷ Laplace at %s, says q found the best Gaussian ",
      "available. Left panel is the two composed\n(%.1f%% total). %s"),
      fmt_med("Laplace ÷ NUTS"), gaussian_gap, fmt_med("VI ÷ Laplace"), vi_gap,
      if (gaussian_gap < 0.5) paste0(
        "With the posterior Gaussian to under half a percent there is nothing here for a ",
        "Gaussian q to get wrong, so the left panel is a statement about the dataset rather ",
        "than a compliment to the estimator."
      ) else sprintf(paste0(
        "The family accounts for it: Laplace alone misses by %.1f%% against q's %.1f%%, so the ",
        "optimizer contributes nothing measurable."), gaussian_gap, vi_gap)),
    x = NULL, y = "ratio"
  ) +
  theme_ferx() + theme(panel.grid.major.x = element_blank())
ggsave(file.path(figs, "anchor-c-why.png"), p3, width = 8.4, height = 5.2, dpi = 200)

cat("read  from", res, "\n")
cat("wrote to  ", figs, "\n")
for (f in c("posterior-overlay.png", "anchor-c-variance.png", "anchor-c-why.png")) {
  fp <- file.path(figs, f)
  cat(sprintf("  %-26s %s\n", f, if (file.exists(fp)) sprintf("ok  %5.0f KB", file.size(fp) / 1024) else "MISSING"))
}
