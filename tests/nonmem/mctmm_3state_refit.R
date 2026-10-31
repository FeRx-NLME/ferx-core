## Independent re-minimisation of the mCTMM anchor likelihood (#2474): R nlminb on the
## exact expm::expm panel likelihood, as a third check on NONMEM's optimum (NONMEM ends
## with ERROR=134 rounding errors after its objective has gone flat).
## Usage: Rscript mctmm_3state_refit.R tests/nonmem/mctmm_3state.csv
## Output (R 4.x, expm): 0.6379289619 -0.4402376070 0.5062269839 1.0645676824 966.4469312412
suppressMessages(library(expm))
d <- read.csv(commandArgs(TRUE)[1]); d <- d[d$MDV == 0, ]
ofv <- function(th) {
  tot <- 0
  for (x in unique(d$X)) {
    tau <- exp(th[1]); a <- c(th[2], th[2] + exp(th[3])); lp <- th[4] * x
    F <- plogis(a - lp); p <- c(F[1], F[2] - F[1], 1 - F[2])
    Q <- matrix(0, 3, 3)
    for (k in 1:2) { dd <- tau * (p[k] + p[k+1]); Q[k,k+1] <- p[k+1]/dd; Q[k+1,k] <- p[k]/dd }
    diag(Q) <- -rowSums(Q)
    s <- d[d$X == x, ]
    for (r in seq_len(nrow(s))) tot <- tot + log(expm(Q * s$DT[r])[s$PSTATE[r] + 1, s$DV[r] + 1])
  }
  -2 * tot
}
f <- nlminb(c(0.5, -0.3, 0.2, 0.3), ofv, control = list(rel.tol = 1e-14, x.tol = 1e-10))
cat(sprintf("%.10f ", c(f$par, f$objective)), "\n")
cat(sprintf("at NONMEM est: %.10f\n", ofv(c(0.637932, -0.440228, 0.506227, 1.06458))))
