args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(6, 8))) {
    stop(paste(
        "usage: fit_cw_cqr.R X.csv Y.csv out_dir tau_csv method r_lib",
        "[maxit tolerance]"
    ))
}

x_path <- args[[1]]
y_path <- args[[2]]
out_dir <- args[[3]]
tau <- as.numeric(strsplit(args[[4]], ",", fixed = TRUE)[[1]])
method <- args[[5]]
r_lib <- args[[6]]
maxit <- if (length(args) == 8) as.integer(args[[7]]) else 1000L
tolerance <- if (length(args) == 8) as.numeric(args[[8]]) else 1e-7

if (nzchar(r_lib)) {
    libraries <- strsplit(r_lib, .Platform$path.sep, fixed = TRUE)[[1]]
    libraries <- libraries[nzchar(libraries)]
    if (length(libraries) > 0) .libPaths(c(libraries, .libPaths()))
}
suppressPackageStartupMessages(library(cqrReg))

X <- as.matrix(read.csv(x_path, check.names = FALSE))
Y <- as.matrix(read.csv(y_path, check.names = FALSE))
storage.mode(X) <- "double"
storage.mode(Y) <- "double"
if (nrow(X) != nrow(Y)) stop("X/Y row mismatch")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
p <- ncol(X)
d <- ncol(Y)
L <- length(tau)
beta <- matrix(NA_real_, nrow = d, ncol = p)
intercepts <- matrix(NA_real_, nrow = d, ncol = L)
runtime <- numeric(d)
objective <- numeric(d)

check_loss <- function(residual, level) {
    residual * (level - as.numeric(residual < 0))
}

for (j in seq_len(d)) {
    started <- proc.time()[[3]]
    fit <- cqr.fit(
        X,
        as.numeric(Y[, j]),
        tau,
        method = method,
        maxit = maxit,
        toler = tolerance
    )
    runtime[[j]] <- proc.time()[[3]] - started
    beta[j, ] <- as.numeric(fit$beta)
    fitted_intercepts <- if (!is.null(fit$intercepts)) fit$intercepts else fit$b
    intercepts[j, ] <- as.numeric(fitted_intercepts)
    base_residual <- as.numeric(Y[, j] - X %*% beta[j, ])
    objective[[j]] <- sum(vapply(
        seq_along(tau),
        function(k) sum(check_loss(base_residual - intercepts[j, k], tau[[k]])),
        numeric(1)
    ))
}

write.csv(beta, file.path(out_dir, "beta.csv"), row.names = FALSE)
write.csv(intercepts, file.path(out_dir, "intercepts.csv"), row.names = FALSE)
write.csv(
    data.frame(response = seq_len(d), runtime_sec = runtime, objective = objective),
    file.path(out_dir, "fit_diagnostics.csv"),
    row.names = FALSE
)
writeLines(
    c(
        paste0("R=", R.version.string),
        paste0("cqrReg=", as.character(packageVersion("cqrReg"))),
        paste0("method=", method),
        paste0("maxit=", maxit),
        paste0("tolerance=", format(tolerance, digits = 17)),
        paste0("tau=", paste(format(tau, digits = 17), collapse = ","))
    ),
    file.path(out_dir, "environment.txt")
)
