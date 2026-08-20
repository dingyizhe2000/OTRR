"""Paper comparison estimators: coordinatewise ASM and MCQR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .baselines import LinearModelFit, fit_ols, fit_zero_intercept_ols


@dataclass
class ASMConfig:
    """Coordinatewise wrapper around the R package ``asm``."""

    betapilot: str = "OLS"
    pilot_coef: np.ndarray | torch.Tensor | None = None
    symmetric: bool = False
    alt_iter: int = 1
    max_iter: int = 65
    intercept_selection: str = "mean"
    center_xy: bool = False
    force_zero_intercept: bool = False
    rscript: str = "Rscript"
    response_workers: int = 1
    numeric_threads: int = 1


@dataclass
class ASMFitInferenceResult:
    """Coordinatewise ASM fit together with the package inference state.

    ``package_covariance`` follows the ``asm`` package convention: it is the
    asymptotic covariance of ``sqrt(n) * (theta_hat - theta_0)``.  The package
    standard errors therefore satisfy ``se**2 = diag(Cov_mat) / n``.
    """

    fit: LinearModelFit
    full_coefficients: torch.Tensor
    internal_intercept: torch.Tensor
    fitted_values: torch.Tensor
    residuals: torch.Tensor
    score_values: torch.Tensor
    information: torch.Tensor
    information_matrices: torch.Tensor
    package_covariance: torch.Tensor
    package_slope_covariance: torch.Tensor
    standard_errors: torch.Tensor
    z_values: torch.Tensor
    p_values: torch.Tensor
    confidence_intervals_95: torch.Tensor
    package_version: str
    converged: tuple[str, ...]

    @property
    def coef(self) -> torch.Tensor:
        return self.fit.coef

    @property
    def x_mean(self) -> torch.Tensor:
        return self.fit.x_mean

    @property
    def y_mean(self) -> torch.Tensor:
        return self.fit.y_mean

    @property
    def method(self) -> str:
        return self.fit.method

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.fit.predict(x)

    def state_dict_for_save(self) -> dict[str, object]:
        return {
            "coef": self.coef.detach().cpu(),
            "x_mean": self.x_mean.detach().cpu(),
            "y_mean": self.y_mean.detach().cpu(),
            "method": self.method,
            "asm_inference": {
                "full_coefficients": self.full_coefficients.detach().cpu(),
                "internal_intercept": self.internal_intercept.detach().cpu(),
                "fitted_values": self.fitted_values.detach().cpu(),
                "residuals": self.residuals.detach().cpu(),
                "score_values": self.score_values.detach().cpu(),
                "information": self.information.detach().cpu(),
                "information_matrices": self.information_matrices.detach().cpu(),
                "package_covariance": self.package_covariance.detach().cpu(),
                "package_slope_covariance": self.package_slope_covariance.detach().cpu(),
                "standard_errors": self.standard_errors.detach().cpu(),
                "z_values": self.z_values.detach().cpu(),
                "p_values": self.p_values.detach().cpu(),
                "confidence_intervals_95": self.confidence_intervals_95.detach().cpu(),
                "package_version": self.package_version,
                "converged": self.converged,
                "covariance_convention": "sqrt_n_asymptotic_covariance",
            },
        }


def _write_csv_tensor(path: Path, x: torch.Tensor) -> None:
    import numpy as np

    np.savetxt(path, x.detach().cpu().numpy(), delimiter=",")


def fit_coordinatewise_asm_with_inference(
    x: torch.Tensor,
    y: torch.Tensor,
    config: ASMConfig | None = None,
) -> ASMFitInferenceResult:
    """Fit coordinatewise ASM and retain the package inference outputs."""

    config = config or ASMConfig()
    if bool(config.center_xy):
        x_fit = x - x.mean(dim=0, keepdim=True)
        y_fit = y - y.mean(dim=0, keepdim=True)
        x_mean = x.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
    else:
        x_fit = x
        y_fit = y
        x_mean = torch.zeros(1, x.shape[1], device=x.device, dtype=x.dtype)
        y_mean = torch.zeros(1, y.shape[1], device=y.device, dtype=y.dtype)

    coef = torch.empty(y.shape[1], x.shape[1], device=x.device, dtype=x.dtype)
    intercept = torch.zeros(1, y.shape[1], device=y.device, dtype=y.dtype)
    with tempfile.TemporaryDirectory(prefix="otrank_asm_") as tmp_raw:
        tmp = Path(tmp_raw)
        x_path = tmp / "x.csv"
        y_path = tmp / "y.csv"
        _write_csv_tensor(x_path, x_fit)
        _write_csv_tensor(y_path, y_fit)
        if config.pilot_coef is None:
            pilot_setup = ""
            pilot_argument = f'"{config.betapilot}"'
        else:
            pilot_values = np.asarray(
                torch.as_tensor(config.pilot_coef).detach().cpu(), dtype=np.float64
            )
            expected_shape = (y.shape[1], x.shape[1])
            if pilot_values.shape != expected_shape:
                raise ValueError(
                    f"ASM pilot_coef must have shape {expected_shape}, got {pilot_values.shape}"
                )
            if not bool(np.isfinite(pilot_values).all()):
                raise ValueError("ASM pilot_coef must contain only finite values")
            pilot_path = tmp / "pilot.csv"
            np.savetxt(pilot_path, pilot_values, delimiter=",")
            pilot_setup = f'Pilot <- as.matrix(read.csv("{pilot_path}", header=FALSE))'
            pilot_argument = "as.numeric(Pilot[j, ])"
        script = f"""
            suppressPackageStartupMessages(library(asm))
            X <- as.matrix(read.csv("{x_path}", header=FALSE))
            Y <- as.matrix(read.csv("{y_path}", header=FALSE))
            {pilot_setup}
            if (is.null(dim(Y))) {{
                Y <- matrix(Y, ncol=1)
            }}
            fit_one <- function(j) {{
                fit <- asm.fit(
                    X, as.numeric(Y[, j]),
                    betapilot={pilot_argument},
                    symmetric={str(config.symmetric).upper()},
                    alt_iter={int(config.alt_iter)},
                    intercept.selection="{config.intercept_selection}",
                    max_iter={int(config.max_iter)},
                    verbose=FALSE
                )
                list(
                    coef=as.numeric(coef(fit)),
                    fitted=as.numeric(fitted(fit)),
                    residuals=as.numeric(residuals(fit)),
                    psi=as.numeric(fit$psi(as.numeric(residuals(fit)))),
                    info=as.numeric(fit$info_asm),
                    I_mat=as.matrix(fit$I_mat),
                    Cov_mat=as.matrix(fit$Cov_mat),
                    std_errs=as.numeric(fit$std_errs),
                    zvals=as.numeric(fit$zvals),
                    sig_vals=as.numeric(fit$sig_vals),
                    ci=as.matrix(confint(fit, level=0.95)),
                    conv=if (is.null(fit$conv) || length(fit$conv) == 0L) "NA" else paste(as.character(fit$conv), collapse=",")
                )
            }}
            columns <- seq_len(ncol(Y))
            workers <- max(1L, min(as.integer({int(config.response_workers)}), length(columns)))
            if (workers > 1L && .Platform$OS.type != "windows") {{
                fits <- parallel::mclapply(columns, fit_one, mc.cores=workers)
            }} else {{
                fits <- lapply(columns, fit_one)
            }}
            stack_rows <- function(values) {{
                widths <- vapply(values, length, integer(1))
                out <- matrix(NA_real_, nrow=length(values), ncol=max(widths))
                for (j in seq_along(values)) {{
                    out[j, seq_along(values[[j]])] <- values[[j]]
                }}
                out
            }}
            write.table(stack_rows(lapply(fits, `[[`, "coef")), file=file.path("{tmp}", "coef.csv"), row.names=FALSE, col.names=FALSE, sep=",")
            write.table(do.call(cbind, lapply(fits, `[[`, "fitted")), file=file.path("{tmp}", "fitted.csv"), row.names=FALSE, col.names=FALSE, sep=",")
            write.table(do.call(cbind, lapply(fits, `[[`, "residuals")), file=file.path("{tmp}", "residuals.csv"), row.names=FALSE, col.names=FALSE, sep=",")
            write.table(do.call(cbind, lapply(fits, `[[`, "psi")), file=file.path("{tmp}", "psi.csv"), row.names=FALSE, col.names=FALSE, sep=",")
            write.table(vapply(fits, `[[`, numeric(1), "info"), file=file.path("{tmp}", "info.csv"), row.names=FALSE, col.names=FALSE, sep=",")
            write.table(stack_rows(lapply(fits, `[[`, "std_errs")), file=file.path("{tmp}", "std_errs.csv"), row.names=FALSE, col.names=FALSE, sep=",")
            write.table(stack_rows(lapply(fits, `[[`, "zvals")), file=file.path("{tmp}", "zvals.csv"), row.names=FALSE, col.names=FALSE, sep=",")
            write.table(stack_rows(lapply(fits, `[[`, "sig_vals")), file=file.path("{tmp}", "sig_vals.csv"), row.names=FALSE, col.names=FALSE, sep=",")
            writeLines(as.character(packageVersion("asm")), file.path("{tmp}", "package_version.txt"))
            writeLines(vapply(fits, `[[`, character(1), "conv"), file.path("{tmp}", "converged.txt"))
            for (j in seq_along(fits)) {{
                tag <- sprintf("%03d", j)
                write.table(fits[[j]]$I_mat, file=file.path("{tmp}", paste0("I_mat_", tag, ".csv")), row.names=FALSE, col.names=FALSE, sep=",")
                write.table(fits[[j]]$Cov_mat, file=file.path("{tmp}", paste0("Cov_mat_", tag, ".csv")), row.names=FALSE, col.names=FALSE, sep=",")
                write.table(fits[[j]]$ci, file=file.path("{tmp}", paste0("ci_", tag, ".csv")), row.names=FALSE, col.names=FALSE, sep=",")
            }}
        """
        env = None
        if int(config.numeric_threads) > 0:
            import os

            env = dict(os.environ)
            thread_value = str(max(1, int(config.numeric_threads)))
            for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                env[name] = thread_value
        completed = subprocess.run(
            [config.rscript, "-e", script],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "R package asm failed. Install it with install.packages('asm') first. "
                f"stderr={completed.stderr.strip()}"
            )
        coef_values = np.loadtxt(tmp / "coef.csv", delimiter=",")
        coef_values = np.asarray(coef_values, dtype=np.float64)
        if coef_values.ndim == 1:
            coef_values = coef_values.reshape(1, -1)
        if coef_values.shape[0] != y.shape[1]:
            raise RuntimeError(f"Expected {y.shape[1]} ASM coefficient rows, got {coef_values.shape[0]}")
        full_width = int(np.count_nonzero(~np.isnan(coef_values[0])))
        full_coefficients = np.empty((y.shape[1], full_width), dtype=np.float64)
        has_intercept = full_width == x.shape[1] + 1
        internal_intercept = np.zeros(y.shape[1], dtype=np.float64)
        for j in range(y.shape[1]):
            row_values = coef_values[j][~np.isnan(coef_values[j])]
            if len(row_values) != full_width:
                raise RuntimeError(
                    "All coordinatewise ASM fits must return the same coefficient width; "
                    f"response 1 returned {full_width}, response {j + 1} returned {len(row_values)}"
                )
            values = torch.as_tensor(
                row_values, device=x.device, dtype=x.dtype
            ).flatten()
            full_coefficients[j] = values.detach().cpu().double().numpy()
            if values.numel() == x.shape[1] + 1:
                intercept[0, j] = values[0]
                internal_intercept[j] = float(values[0].item())
                values = values[-x.shape[1] :]
            if values.numel() != x.shape[1]:
                raise RuntimeError(f"Unexpected asm coefficient length {values.numel()} for p={x.shape[1]}")
            coef[j] = values

        def load_matrix(name: str, rows: int, cols: int) -> np.ndarray:
            value = np.asarray(np.loadtxt(tmp / name, delimiter=","), dtype=np.float64)
            if value.size != rows * cols:
                raise RuntimeError(
                    f"Expected {name} to contain {rows * cols} entries, got {value.size}"
                )
            return value.reshape(rows, cols)

        n_obs = int(x.shape[0])
        n_responses = int(y.shape[1])
        fitted_values = load_matrix("fitted.csv", n_obs, n_responses)
        residuals = load_matrix("residuals.csv", n_obs, n_responses)
        score_values = load_matrix("psi.csv", n_obs, n_responses)
        information = np.atleast_1d(np.loadtxt(tmp / "info.csv", delimiter=",")).astype(np.float64)
        if information.size != n_responses:
            raise RuntimeError(
                f"Expected {n_responses} ASM information values, got {information.size}"
            )
        standard_errors = load_matrix("std_errs.csv", n_responses, full_width)
        z_values = load_matrix("zvals.csv", n_responses, full_width)
        p_values = load_matrix("sig_vals.csv", n_responses, full_width)
        information_matrices = []
        package_covariance = []
        confidence_intervals = []
        for j in range(y.shape[1]):
            tag = f"{j + 1:03d}"
            information_matrices.append(
                load_matrix(f"I_mat_{tag}.csv", x.shape[1], x.shape[1])
            )
            package_covariance.append(
                load_matrix(f"Cov_mat_{tag}.csv", full_width, full_width)
            )
            confidence_intervals.append(
                load_matrix(f"ci_{tag}.csv", full_width, 2)
            )
        information_matrices_np = np.stack(information_matrices)
        package_covariance_np = np.stack(package_covariance)
        confidence_intervals_np = np.stack(confidence_intervals)
        slope_slice = slice(1, None) if has_intercept else slice(None)
        package_slope_covariance_np = package_covariance_np[:, slope_slice, slope_slice]
        package_version = (tmp / "package_version.txt").read_text().strip()
        converged = tuple((tmp / "converged.txt").read_text().splitlines())

    if bool(config.force_zero_intercept):
        y_mean = torch.zeros_like(y_mean)
    elif bool(config.center_xy):
        y_mean = y_mean + intercept
    else:
        y_mean = intercept
    fit = LinearModelFit(coef=coef, x_mean=x_mean, y_mean=y_mean, method="coordinatewise_asm")
    return ASMFitInferenceResult(
        fit=fit,
        full_coefficients=torch.as_tensor(full_coefficients, dtype=torch.float64),
        internal_intercept=torch.as_tensor(internal_intercept, dtype=torch.float64),
        fitted_values=torch.as_tensor(fitted_values, dtype=torch.float64),
        residuals=torch.as_tensor(residuals, dtype=torch.float64),
        score_values=torch.as_tensor(score_values, dtype=torch.float64),
        information=torch.as_tensor(information, dtype=torch.float64),
        information_matrices=torch.as_tensor(information_matrices_np, dtype=torch.float64),
        package_covariance=torch.as_tensor(package_covariance_np, dtype=torch.float64),
        package_slope_covariance=torch.as_tensor(package_slope_covariance_np, dtype=torch.float64),
        standard_errors=torch.as_tensor(standard_errors, dtype=torch.float64),
        z_values=torch.as_tensor(z_values, dtype=torch.float64),
        p_values=torch.as_tensor(p_values, dtype=torch.float64),
        confidence_intervals_95=torch.as_tensor(confidence_intervals_np, dtype=torch.float64),
        package_version=package_version,
        converged=converged,
    )


def fit_coordinatewise_asm(
    x: torch.Tensor,
    y: torch.Tensor,
    config: ASMConfig | None = None,
) -> LinearModelFit:
    """Backward-compatible point-estimation interface for coordinatewise ASM."""

    return fit_coordinatewise_asm_with_inference(x, y, config).fit


@dataclass
class MCQROTConfig:
    """Approximate multiple-output CQR through the empirical OT lens."""

    max_iter: int = 200
    lr: float = 5e-2
    ridge: float = 1e-5
    reference: str = "normal"
    reference_values: np.ndarray | torch.Tensor | None = None
    seed: int = 2026
    show_progress: bool = False
    intercept_mode: str = "zero"
    optimizer: str = "adam"


def _sample_mcqr_reference(n: int, d: int, generator: torch.Generator, x: torch.Tensor, reference: str) -> torch.Tensor:
    reference_name = str(reference).lower()
    if reference_name in {"normal", "gaussian"}:
        return torch.randn(n, d, generator=generator, device=x.device, dtype=x.dtype)
    if reference_name in {"hypercube", "uniform_hypercube"}:
        return torch.rand(n, d, generator=generator, device=x.device, dtype=x.dtype)
    if reference_name == "uniform_ball":
        from otrank.ot_maps.reference import sample_uniform_ball

        return sample_uniform_ball(n, d, generator, device=x.device, dtype=x.dtype)
    raise ValueError("MCQR reference must be 'normal', 'hypercube', or 'uniform_ball'")


def fit_mcqr_ot_lens(x: torch.Tensor, y: torch.Tensor, config: MCQROTConfig | None = None) -> LinearModelFit:
    """Fit a linear MCQR/OT-lens estimator by subgradient-style optimization."""

    config = config or MCQROTConfig()
    if config.intercept_mode.lower() == "zero":
        pilot = fit_zero_intercept_ols(x, y, ridge=1e-6)
    elif config.intercept_mode.lower() in {"center", "intercept"}:
        pilot = fit_ols(x, y, ridge=1e-6)
    else:
        raise ValueError("MCQROTConfig.intercept_mode must be 'zero', 'center', or 'intercept'")
    xc = x - pilot.x_mean
    yc = y - pilot.y_mean
    coef = torch.nn.Parameter(pilot.coef.detach().clone())
    generator = torch.Generator(device=str(x.device))
    generator.manual_seed(int(config.seed))
    if config.reference_values is None:
        reference = _sample_mcqr_reference(x.shape[0], y.shape[1], generator, x, config.reference)
    else:
        reference = torch.as_tensor(config.reference_values, device=x.device, dtype=x.dtype)
        expected_shape = (x.shape[0], y.shape[1])
        if tuple(reference.shape) != expected_shape:
            raise ValueError(
                f"MCQR reference_values must have shape {expected_shape}, got {tuple(reference.shape)}"
            )
        if not bool(torch.isfinite(reference).all()):
            raise ValueError("MCQR reference_values must contain only finite values")
    optimizer_name = config.optimizer.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam([coef], lr=float(config.lr))
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD([coef], lr=float(config.lr))
    else:
        raise ValueError("MCQROTConfig.optimizer must be 'adam' or 'sgd'")

    for _ in range(max(1, int(config.max_iter))):
        with torch.no_grad():
            residuals = yc - xc @ coef.T
            score = residuals @ reference.T
            _, col_ind = linear_sum_assignment((-score).detach().cpu().numpy())
            matched = reference[torch.as_tensor(col_ind, dtype=torch.long, device=x.device)]
        optimizer.zero_grad()
        residuals = yc - xc @ coef.T
        loss = torch.sum(residuals * matched, dim=1).mean()
        if config.ridge > 0:
            loss = loss + float(config.ridge) * torch.sum(coef * coef)
        loss.backward()
        optimizer.step()

    return LinearModelFit(coef=coef.detach(), x_mean=pilot.x_mean, y_mean=pilot.y_mean, method="mcqr_ot_lens")
