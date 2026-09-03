# SEM (OpenMx) -- classical twin multigroup ACE, ML on the per-DOR observed
# covariance matrix: cov(y1,y2) = w_G^d sigma_A^2 + c_d sigma_C^2.
# Input: per-DOR 2x2 (c11_d,c12_d,c22_d) + Nd_d (from sem_summary() in generate.py).
# Any other column is an id column and is passed through unchanged.
# Batches all rows through one Rscript call (OpenMx startup ~2s, amortized).
#
# usage: Rscript sem.R sem_in.csv out.csv
suppressMessages(library(OpenMx))
options(warn = -1)
mxOption(NULL, "Default optimizer", "SLSQP")

args <- commandArgs(trailingOnly = TRUE)
want_se <- length(args) >= 3 && args[3] == "se"
mxOption(NULL, "Calculate Hessian", if (want_se) "Yes" else "No")
mxOption(NULL, "Standard Errors", if (want_se) "Yes" else "No")

REL <- c(0.5, 0.25, 0.125)                       # w_G^d (fixed)
CONDS <- list(AE = c(0, 0, 0), step = c(1, 0, 0), const = c(1, 1, 1))

fit_ace <- function(s11, s12, s22, Nd, cvec) {
  grp <- function(d) {
    S <- matrix(c(s11[d], s12[d], s12[d], s22[d]), 2,
                dimnames = list(c("y1", "y2"), c("y1", "y2")))
    off <- sprintf("%g*ace.A + %g*ace.C", REL[d], cvec[d])
    expr <- sprintf("rbind(cbind(ace.V, %s), cbind(%s, ace.V))", off, off)
    mxModel(paste0("d", d),
      mxAlgebraFromString(expr, name = "EC",
                          dimnames = list(c("y1", "y2"), c("y1", "y2"))),
      mxExpectationNormal("EC"), mxFitFunctionML(),
      mxData(S, type = "cov", numObs = Nd[d]))
  }
  free_C <- any(cvec != 0)
  v0 <- mean(c(s11, s22))
  top <- mxModel("ace",
    mxMatrix("Full", 1, 1, TRUE, 0.4 * v0, "a", name = "A"),
    mxMatrix("Full", 1, 1, free_C, if (free_C) 0.1 * v0 else 0, "c", name = "C"),
    mxMatrix("Full", 1, 1, TRUE, 0.3 * v0, "e", name = "E"),
    mxAlgebra(A + C + E, name = "V"),
    grp(1), grp(2), grp(3),
    mxFitFunctionMultigroup(c("d1", "d2", "d3")))
  f <- try(mxRun(top, silent = TRUE, suppressWarnings = TRUE), silent = TRUE)
  if (inherits(f, "try-error")) return(c(NA, NA, NA, NA, NA))
  A <- mxEval(A, f); C <- mxEval(C, f); E <- mxEval(E, f); v <- A + C + E
  vc <- if (free_C) C / v else NA
  se_g <- NA; se_s <- NA
  if (want_se) {
    se_g <- tryCatch(as.numeric(mxSE(A / V, f, silent = TRUE)), error = function(e) NA)
    if (free_C)
      se_s <- tryCatch(as.numeric(mxSE(C / V, f, silent = TRUE)), error = function(e) NA)
  }
  c(A / v, vc, f$output$status$code, se_g, se_s)
}

ev <- read.csv(args[1], stringsAsFactors = FALSE)
covcols <- grep("^(c11|c12|c22|Nd)_[123]$", names(ev), value = TRUE)
idcols  <- setdiff(names(ev), covcols)
out <- list()
for (i in seq_len(nrow(ev))) {
  s11 <- c(ev$c11_1[i], ev$c11_2[i], ev$c11_3[i])
  s12 <- c(ev$c12_1[i], ev$c12_2[i], ev$c12_3[i])
  s22 <- c(ev$c22_1[i], ev$c22_2[i], ev$c22_3[i])
  Nd  <- c(ev$Nd_1[i],  ev$Nd_2[i],  ev$Nd_3[i])
  for (cn in names(CONDS)) {
    r <- fit_ace(s11, s12, s22, Nd, CONDS[[cn]])
    res <- data.frame(condition = cn, V_G = r[1], V_S = r[2], status = r[3])
    if (want_se) { res$se_VG <- r[4]; res$se_VS <- r[5] }
    out[[length(out) + 1]] <- cbind(ev[i, idcols, drop = FALSE], res, row.names = NULL)
  }
}
write.csv(do.call(rbind, out), args[2], row.names = FALSE)
