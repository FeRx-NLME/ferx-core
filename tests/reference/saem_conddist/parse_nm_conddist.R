# parse_nm_conddist.R — fetch the existing NONMEM warfarin.phi/.ext from the
# SAEM+IMP run and compare its conditional distribution to ferx (#257).
#
# NONMEM EM .phi columns: PHI(k) = conditional mean of phi_k = MU_k + eta_k
# (= log P_i with MU-referencing); PHC(k,k) = conditional variance of eta_k.
# eta conditional mean = PHI(k) - log(THETA_k); conditional SD = sqrt(PHC(k,k)).
# We use the LAST table in the .phi (METHOD=IMP EONLY moments).
#
# Run from the repo root after:
#   cargo run --release -p ferx-cli -- examples/warfarin_saem_conddist.ferx --data data/warfarin.csv
# Then (requires NM_PASS env var with the klebsiella SSH password):
#   NM_PASS=<pw> Rscript tests/reference/saem_conddist/parse_nm_conddist.R

NM_HOST <- "klebsiella.lacdr.leidenuniv.nl"; NM_USER <- "beekh"
NM_PASS <- Sys.getenv("NM_PASS", ""); NM_DIR <- "/home/beekh/nm_saem_conddist_257"
FERX_CD <- "warfarin_saem_conddist-conddist.csv"
OUT     <- "bench_conddist_nonmem.csv"
if (nchar(NM_PASS) == 0) stop("NM_PASS not set")

ASK <- "/tmp/ferx_askpass.sh"
writeLines(c("#!/bin/sh", paste0('echo "', NM_PASS, '"')), ASK)
system(paste("chmod +x", ASK), ignore.stdout = TRUE, ignore.stderr = TRUE)
envp <- function() sprintf("SSH_ASKPASS='%s' SSH_ASKPASS_REQUIRE=force", ASK)
scp_from <- function(r, l) system(sprintf("%s scp -o StrictHostKeyChecking=no %s@%s:'%s' '%s'", envp(), NM_USER, NM_HOST, r, l), intern = TRUE)

phi_l <- tempfile(fileext = ".phi"); ext_l <- tempfile(fileext = ".ext")
scp_from(file.path(NM_DIR, "warfarin.phi"), phi_l)
scp_from(file.path(NM_DIR, "warfarin.ext"), ext_l)

## ---- final THETA from .ext (last table, final-estimate row -1000000000) ----
el <- readLines(ext_l)
estart <- tail(grep("^TABLE NO", el), 1)
ehdr   <- grep("^\\s*ITERATION", el); ehdr <- ehdr[ehdr > estart][1]
ecol   <- strsplit(trimws(el[ehdr]), "\\s+")[[1]]
erow   <- grep("^\\s*-1000000000\\b", el); erow <- erow[erow > estart][1]
ev     <- setNames(as.numeric(strsplit(trimws(el[erow]), "\\s+")[[1]]), ecol)
theta  <- c(ev[["THETA1"]], ev[["THETA2"]], ev[["THETA3"]])
cat("NONMEM final THETA (TVCL,TVV,TVKA):", paste(round(theta, 4), collapse = ", "), "\n")

## ---- LAST .phi table (IMP EONLY conditional moments) ----
pl <- readLines(phi_l)
pstart <- tail(grep("^TABLE NO", pl), 1)
phdr   <- grep("PHI\\(1\\)", pl); phdr <- phdr[phdr > pstart][1]
pcol   <- strsplit(trimws(pl[phdr]), "\\s+")[[1]]
body   <- pl[(phdr + 1):length(pl)]; body <- body[grepl("[0-9]", body)]
phi <- read.table(text = paste(body, collapse = "\n"), col.names = pcol,
                  check.names = FALSE, stringsAsFactors = FALSE)

mu <- log(theta)
nm <- data.frame(
  ID      = rep(phi[["ID"]], 3),
  param   = rep(c("CL", "V", "KA"), each = nrow(phi)),
  nm_mean = c(phi[["PHI(1)"]] - mu[1], phi[["PHI(2)"]] - mu[2], phi[["PHI(3)"]] - mu[3]),
  nm_sd   = c(sqrt(phi[["PHC(1,1)"]]), sqrt(phi[["PHC(2,2)"]]), sqrt(phi[["PHC(3,3)"]])),
  stringsAsFactors = FALSE)

## ---- ferx ----
ferx <- read.csv(FERX_CD, stringsAsFactors = FALSE)
ferx$param <- sub("^ETA_", "", ferx$ETA)
cmp <- merge(ferx, nm, by = c("ID", "param")); cmp <- cmp[order(cmp$param, cmp$ID), ]

summ <- function(a, b, lbl) cat(sprintf("%-12s  corr=%.4f  max|diff|=%.4f  rmse=%.4f\n",
  lbl, suppressWarnings(cor(a, b)), max(abs(a - b)), sqrt(mean((a - b)^2))))
cat("\n=== ferx vs NONMEM (per-subject eta, conditional distribution) ===\n")
summ(cmp$COND_MEAN, cmp$nm_mean, "cond mean")
summ(cmp$COND_SD,   cmp$nm_sd,   "cond sd")
summ(cmp$COND_MODE, cmp$nm_mean, "mode vs nm mean")

write.csv(cmp, OUT, row.names = FALSE)
cat("\nWrote comparison to", OUT, "\n")
print(head(cmp[, c("ID","param","COND_MEAN","nm_mean","COND_SD","nm_sd")], 6), row.names = FALSE)
