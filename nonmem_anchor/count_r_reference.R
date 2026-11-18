# External references for the #2877 count anchors (run from nonmem_anchor/):
#   Rscript count_r_reference.R
# glm / MASS::glm.nb for the fixed-effects case, lme4::glmer(nAGQ = 1, Laplace) /
# lme4::glmer.nb for the random-intercept case. Prints full-precision estimates and
# -2 logLik (the scale of the ferx OFV, ln Γ(k+1) included).
suppressMessages({ library(lme4); library(MASS) })
options(digits = 15)
show <- function(tag, fit, extra = NULL) {
  cat(tag, "\n"); print(fixef_or_coef(fit)); if (!is.null(extra)) print(extra)
  cat("m2ll", sprintf("%.10f", -2 * as.numeric(logLik(fit))), "\n\n")
}
fixef_or_coef <- function(f) if (inherits(f, "merMod")) fixef(f) else coef(f)
for (f in c("count_pois.csv", "count_nb.csv")) {
  d <- read.csv(f); d$W <- d$WT / 70 - 1
  cat("=====", f, "\n")
  show("glm poisson", glm(DV ~ W + TRT, family = poisson, data = d))
  nb <- glm.nb(DV ~ W + TRT, data = d, control = glm.control(epsilon = 1e-12, maxit = 200))
  show("glm.nb", nb, c(theta = nb$theta))
  ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  gp <- glmer(DV ~ W + TRT + (1 | ID), family = poisson, data = d, nAGQ = 1, control = ctl)
  show("glmer poisson", gp, c(omega = VarCorr(gp)$ID[1]))
  gn <- glmer.nb(DV ~ W + TRT + (1 | ID), data = d, nAGQ = 1, control = ctl)
  show("glmer.nb", gn, c(omega = VarCorr(gn)$ID[1], r = getME(gn, "glmer.nb.theta")))
}
