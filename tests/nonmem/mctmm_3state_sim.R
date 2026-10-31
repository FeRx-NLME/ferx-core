## Simulate a 3-state mCTMM (fixed effects) and write the ferx + NONMEM datasets (#2474).
## Also checks the Sylvester closed form used in mctmm_3state.ctl against expm::expm.
## Usage: Rscript mctmm_3state_sim.R <outdir>
##   -> <outdir>/mctmm_3state.csv (NONMEM; committed here) and
##      <outdir>/mctmm_3state_ferx.csv (ferx; committed as data/mctmm_3state.csv).
suppressMessages(library(expm))
args <- commandArgs(TRUE); out <- args[1]
set.seed(2474)
LTAU <- log(2); C1 <- -0.5; LD2 <- log(1.5); BX <- 0.8
gen <- function(x) {
  tau <- exp(LTAU); a <- c(C1, C1 + exp(LD2)); lp <- BX * x
  F <- plogis(a - lp); p <- c(F[1], F[2] - F[1], 1 - F[2])
  Q <- matrix(0, 3, 3)
  for (k in 1:2) {
    d <- tau * (p[k] + p[k + 1])
    Q[k, k + 1] <- p[k + 1] / d; Q[k + 1, k] <- p[k] / d
  }
  diag(Q) <- -rowSums(Q)
  list(Q = Q, p = p)
}
## Sylvester closed form (the NONMEM $PRED construction) vs expm.
syl <- function(Q, t) {
  A <- Q[1,2]; B <- Q[2,1]; C <- Q[2,3]; D <- Q[3,2]
  S <- A + B + C + D; M <- A*C + A*D + B*D
  disc <- sqrt(S*S - 4*M); L1 <- (-S + disc)/2; L2 <- (-S - disc)/2
  Q2 <- Q %*% Q; I <- diag(3)
  (Q2 + S*Q + M*I)/M + exp(L1*t)*(Q2 - L2*Q)/(L1*(L1-L2)) + exp(L2*t)*(Q2 - L1*Q)/(L2*(L2-L1))
}
g <- gen(1)
stopifnot(max(abs(g$p %*% g$Q)) < 1e-14)            # π is stationary
for (t in c(0.1, 1, 3.7)) stopifnot(max(abs(syl(g$Q, t) - expm(g$Q * t))) < 1e-12)
cat("closed form OK; pi =", g$p, "\n")

nsub <- 80
fx <- NULL; nm <- NULL
for (i in 1:nsub) {
  x <- as.numeric(i %% 2 == 0)
  g <- gen(x)
  s <- sample(0:2, 1, prob = g$p)
  t <- 0; nobs <- sample(6:12, 1)
  fx <- rbind(fx, data.frame(ID = i, TIME = 0, DV = s, CMT = 5, EVID = 0, MDV = 0, X = x))
  nm <- rbind(nm, data.frame(ID = i, DV = s, PSTATE = s, DT = 0, X = x, MDV = 1))
  for (j in 2:nobs) {
    dt <- round(runif(1, 0.2, 3), 2)
    P <- expm(g$Q * dt)
    s2 <- sample(0:2, 1, prob = pmax(P[s + 1, ], 0))
    t <- t + dt
    fx <- rbind(fx, data.frame(ID = i, TIME = t, DV = s2, CMT = 5, EVID = 0, MDV = 0, X = x))
    nm <- rbind(nm, data.frame(ID = i, DV = s2, PSTATE = s, DT = dt, X = x, MDV = 0))
    s <- s2
  }
}
write.csv(fx, file.path(out, "mctmm_3state_ferx.csv"), row.names = FALSE, quote = FALSE)
write.csv(nm, file.path(out, "mctmm_3state.csv"), row.names = FALSE, quote = FALSE)
cat("rows", nrow(fx), "transitions", sum(nm$MDV == 0), "\n")
print(table(fx$DV))
