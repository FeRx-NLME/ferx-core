# Benchmark: ferx SAEM conditional-distribution pass vs saemix conddist.saemix
# (#257). Same warfarin data, same 1-cpt oral log-normal model, proportional
# error. Compares per-subject conditional mean / SD / mode of the random
# effects (eta) on the log scale.
#
# Run from the repo root after:
#   cargo run --release -p ferx-cli -- examples/warfarin_saem_conddist.ferx --data data/warfarin.csv
# Then:
#   Rscript tests/reference/saem_conddist/bench_saemix_conddist.R
#
# Requires: saemix  (install.packages("saemix"))

suppressMessages(library(saemix))

DATA    <- "data/warfarin.csv"
FERX_CD <- "warfarin_saem_conddist-conddist.csv"
OUT     <- "bench_conddist_comparison.csv"

## ---- 1. ferx conditional distribution (already computed by the CLI) ----
ferx <- read.csv(FERX_CD, stringsAsFactors = FALSE)
# columns: ID, ETA, COND_MEAN, COND_SD, COND_MODE
ferx$param <- sub("^ETA_", "", ferx$ETA)   # CL / V / KA
cat("ferx conddist rows:", nrow(ferx), "\n")

## ---- 2. saemix fit on the same data ----
raw <- read.csv(DATA, na.strings = ".", stringsAsFactors = FALSE)
obs <- subset(raw, EVID == 0 & MDV == 0)
obs <- obs[!is.na(obs$DV), c("ID", "TIME", "DV")]

sdata <- saemixData(
  name.data       = obs,
  name.group      = "ID",
  name.predictors = "TIME",
  name.response   = "DV"
)

# 1-cpt oral, single 100 mg dose at t=0 (warfarin data: AMT=100 for every ID).
model1cpt <- function(psi, id, xidep) {
  tim <- xidep[, 1]
  dose <- 100
  CL <- psi[id, 1]; V <- psi[id, 2]; ka <- psi[id, 3]
  k <- CL / V
  ypred <- dose * ka / (V * (ka - k)) * (exp(-k * tim) - exp(-ka * tim))
  ypred
}

smodel <- saemixModel(
  model            = model1cpt,
  description      = "1-cpt oral",
  psi0             = matrix(c(0.2, 10.0, 1.5), ncol = 3,
                            dimnames = list(NULL, c("CL", "V", "ka"))),
  transform.par    = c(1, 1, 1),                 # log-normal (matches ferx exp(eta))
  covariance.model = diag(3),                    # diagonal omega
  omega.init       = diag(c(0.09, 0.04, 0.30)),
  error.model      = "proportional"
)

sopt <- list(
  seed           = 12345,
  nbiter.saemix  = c(300, 400),                  # match ferx explore/converge
  displayProgress = FALSE,
  save           = FALSE,
  save.graphs    = FALSE,
  print          = FALSE
)

fit <- saemix(smodel, sdata, sopt)

# Conditional distribution (MCMC) and conditional mode (MAP).
fit <- conddist.saemix(fit, nsamp = 1)
fit <- map.saemix(fit)

## ---- 3. assemble saemix per-subject eta (log scale) ----
pop_psi <- fit@results@fixed.effects          # CL, V, ka population estimates
mu_phi  <- log(pop_psi)                        # phi = log(psi); mu = log(theta_hat)
ids     <- fit@data@data[[fit@data@name.group]]
uid     <- unique(ids)

cond_phi <- fit@results@cond.mean.phi          # E[phi_i | y_i]
cond_sd  <- sqrt(fit@results@cond.var.phi)      # SD[phi_i | y_i] = SD[eta_i]
map_phi  <- fit@results@map.phi                 # MAP phi (mode)

pnames <- c("CL", "V", "KA")
sae <- do.call(rbind, lapply(seq_along(uid), function(i) {
  data.frame(
    ID        = uid[i],
    param     = pnames,
    sae_mean  = as.numeric(cond_phi[i, ]) - mu_phi,   # eta cond mean
    sae_sd    = as.numeric(cond_sd[i, ]),
    sae_mode  = as.numeric(map_phi[i, ])  - mu_phi,   # eta mode
    stringsAsFactors = FALSE
  )
}))

cat("\nPopulation parameters (saemix):\n"); print(round(pop_psi, 4))

## ---- 4. merge and compare ----
cmp <- merge(ferx, sae, by = c("ID", "param"))
cmp <- cmp[order(cmp$param, cmp$ID), ]

summ <- function(a, b, lbl) {
  d <- a - b
  cat(sprintf("%-10s  corr=%.4f  max|diff|=%.4f  rmse=%.4f\n",
              lbl, suppressWarnings(cor(a, b)), max(abs(d)), sqrt(mean(d^2))))
}
cat("\n=== ferx vs saemix (per-subject eta) ===\n")
summ(cmp$COND_MEAN, cmp$sae_mean, "cond mean")
summ(cmp$COND_SD,   cmp$sae_sd,   "cond sd")
summ(cmp$COND_MODE, cmp$sae_mode, "mode/MAP")

write.csv(cmp, OUT, row.names = FALSE)
cat("\nWrote comparison to", OUT, "\n")
