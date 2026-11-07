# Optimal Transport Rank Regression for Robust Multivariate-Response Regression

This repository contains reproducibility code for the paper
"Optimal Transport Rank Regression for Robust Multivariate-Response
Regression." It includes the optimal transport rank regression
(OTRR) estimators, the linear and ReLU simulation workflows, inference
utilities, and the NHANES oral-microbiome composition analysis.

## Repository Layout

The main entry points and directories are:

- `run_simulation_study.py`: user-facing simulation reproduction entry point.
- `run_real_data_experiment.py`: user-facing real-data reproduction entry point.
- `reproduce_paper_outputs.py`: builds tables and figures from the results.
- `notebooks/reproduce_paper_results.ipynb`: displays the paper tables and
  figures.
- `configs/simulation_linear.json`: final linear-simulation configuration.
- `configs/simulation_relu.json`: final ReLU-simulation configuration.
- `configs/main_real_data.json`: real-data application configuration.
- `src/otrank/`: reusable OTRR package code.
- `src/otrank/experiments/`: simulation execution, scheduling, checkpointing,
  ensemble aggregation, and monitoring.
- `src/otrank/inference/`: IF, sandwich, Monte Carlo, and covariance summaries.
- `src/otrank/real_data/`: NHANES data preparation and baseline utilities.

## Software Requirements

The package requires Python 3.10 or newer. Install the package and Python
dependencies in an environment of your choice:

```bash
pip install -e .
# Optional notebook environment
pip install -e '.[paper]'
# Development and test environment
pip install -e '.[test]'
```

The reported GPU experiments were run using the following software
environment:

| Component | Version |
|---|---|
| Operating system | Ubuntu 24.04.2 LTS |
| Python | 3.11.15 |
| PyTorch | 2.1.2+cu121 |
| CUDA runtime used by PyTorch | 12.1 |
| Triton | 2.1.0 |
| NumPy | 1.25.0 |
| SciPy | 1.10.1 |
| pandas | 1.5.3 |

The primary workstation used an AMD EPYC 7532 CPU, 512 GB RAM, and an NVIDIA
RTX 3090 24 GB GPU. Some linear-simulation jobs used NVIDIA A100 PCIe
40 GB GPUs.

The retained baseline workflows use `asm` 0.2.4 and `cqrReg` 1.2.1.
The portable frozen-output pipeline has also been checked with Python 3.12.4,
PyTorch 2.12.0, NumPy 2.2.6, SciPy 1.17.1, pandas 2.3.3 and Matplotlib 3.10.9.
This is a postprocessing environment, not a claim of bitwise GPU-fit replay.

The ASM and CW-CQR baselines require R and the R packages `asm` and `cqrReg`. Install R
and then run:

```bash
Rscript -e 'install.packages(c("asm", "cqrReg"), repos="https://cloud.r-project.org")'
```

If `cqrReg` is installed in a nonstandard user library, set `CQR_R_LIB` to
that directory. Otherwise, the bridge uses R's normal `.libPaths()`, including
`R_LIBS_USER`.

## Reproduce the Paper Results

Run the commands below from the repository root. The included `paper_data/`
contains versioned simulation-repetition metrics and aggregate application
results only. Default generation needs no GPU, R, private download, raw
participant data, estimator training or OT solve. It selects the current
centered plug-in inference and coefficient-equivariance results.

```bash
python reproduce_paper_outputs.py --project-root .
```

Use `--output-root <directory>` for a separate generated-results directory, or
`--skip-build` to validate and index existing outputs without regenerating them.
The notebook uses the same pipeline. `results/paper_output_index.json` uses
semantic keys independent of manuscript table/figure numbering.

Large historical fitted-state archives are optional private downloads.
`paper_artifacts.json` identifies their scope explicitly: the old archive hashes
do not include the later centering and coefficient-equivariance updates.
`download_paper_artifacts.py` fails until an authorized host is configured; no
download URL or new release is implied.

### Linear simulation

Regenerate the linear tables and boxplots, including interval lengths,
coverages and entrywise coefficient equivariance errors, from all 50 retained
repetitions per method/noise:

```bash
python reproduce_paper_outputs.py --project-root . --only linear
```

The manifest records pairing provenance, including MCQR's pooled historical and
supplemental CPU pairs. Plots use actual observations, not display thresholds.
To rerun estimator training instead:

```bash
python run_simulation_study.py --config configs/simulation_linear.json \
  --out-root reproduced_results/linear
```

Fresh fits are separate from frozen paper-output reproduction. This training
command does not run the full K=50 covariance study or import new fits into the
paper data. A matching-data inference/analysis step is required; generic API
solver defaults are not the paper's numerical protocol.

For maintainers with full retained IF bundles predating the covariance correction,
rebuild the 150 linear OTRR inference artifacts without refitting or re-solving OT:

```bash
PYTHONPATH=src python -m otrank.results_analysis.recenter_linear_inference \
  --project-root . --activate
PYTHONPATH=src python -m otrank.results_analysis.finalize_linear_simulation \
  --project-root . --inference-only
```

This creates the immutable `inference_versions/centered_plugin_v1` directory,
preserves the original artifacts and paper outputs, and records the active
version in `config/active_inference.json`. The migration exploits the retained
common-design Kronecker IF structure, with an explicit numerical reconstruction
tolerance; it does not reproduce the original GPU random clouds. The retained
finalizer uses the selected version; the default compact pipeline uses already
corrected CSVs. A completed
version is not overwritten by rerunning the migration.

### ReLU simulation

Regenerate the ReLU prediction table from the 50 retained repetitions per
method/noise. To retrain instead, use the separate config-driven command:

```bash
python reproduce_paper_outputs.py --project-root . --only relu
# Estimator training, not needed for frozen-result reproduction:
python run_simulation_study.py --config configs/simulation_relu.json \
  --out-root reproduced_results/relu
```

### NHANES real-data application

Regenerate the NHANES table, pairwise stability and supplementary semi-synthetic
table from aggregate/fold-level metrics. No participant records are included.

```bash
python reproduce_paper_outputs.py --project-root . --only nhanes
```

`run_real_data_experiment.py` also defaults to the compact frozen metrics.
Its explicit `--retained-source` option requires the full privately retained
states; it does not constitute a one-command full NHANES OTRR refit workflow.

The supplementary NHANES semi-synthetic estimator experiment is a separate,
resumable estimator rerun driven by its own authoritative JSON config:

```bash
PYTHONPATH=src python run_nhanes_semisynthetic_invariance.py \
  --config configs/real_data_nhanes_semisynthetic_invariance.json
```

Use `--r-lib` to override the CW-CQR R library for this command. Otherwise the
runner checks `CQR_R_LIB`, then `R_LIBS_USER`, and finally R's default
`.libPaths()`.

## Fit OTRR on a New Dataset

Both examples expect predictor and multivariate-response arrays with shapes
`(n, p)` and `(n, d)`. Use `dense_triton` on a CUDA device or
`dense_pytorch` for a portable CPU implementation.

### Linear model

The linear API below estimates the training means, fits a 20-level CW-CQR pilot
to the centered data, and then uses its common slope matrix to initialize one
OTRR reference member. The fitted object centers new predictors using the
training predictor mean and restores the training response mean.

```python
from pathlib import Path
import torch

from otrank import LinearOTRankConfig, as_tensor_pair, fit_ot_rank_linear
from otrank.estimators import CWCQRFitConfig, LinearModelFit, fit_cw_cqr

x, y = as_tensor_pair(X, Y, device="cuda", dtype=torch.float32)
x_mean = x.mean(dim=0, keepdim=True)
y_mean = y.mean(dim=0, keepdim=True)
x_centered = x - x_mean
y_centered = y - y_mean

cqr = fit_cw_cqr(
    x_centered.detach().cpu().double().numpy(),
    y_centered.detach().cpu().double().numpy(),
    output_dir=Path("cw_cqr_initializer"),
    config=CWCQRFitConfig(),
)

pilot = LinearModelFit(
    coef=torch.as_tensor(cqr.beta, device=x.device, dtype=x.dtype),
    x_mean=x_mean,
    y_mean=y_mean,
    method="cw_cqr",
)

fit = fit_ot_rank_linear(
    x,
    y,
    init_fit=pilot,
    config=LinearOTRankConfig(
        backend="dense_triton",
        reference_seed=20260721,
        intercept_mode="pilot",
    ),
)

x_new = torch.as_tensor(X_new, device=x.device, dtype=x.dtype)
y_pred = fit.predict(x_new)
```

### ReLU network

The ReLU convenience API first fits a CW-CQR network and then refines the same
network by OTRR. Network width, depth, optimization settings, and reference
size are controlled by `ReLUOTRankConfig`.

```python
import torch
from otrank import ReLUOTRankConfig, as_tensor_pair, fit_ot_rank_relu

x, y = as_tensor_pair(X, Y, device="cuda", dtype=torch.float32)
fit = fit_ot_rank_relu(
    x,
    y,
    config=ReLUOTRankConfig(backend="dense_triton", reference_seed=20260721),
)
y_pred = fit.predict(torch.as_tensor(X_new, device=x.device, dtype=x.dtype))
```

For a `K`-reference ensemble, fit the members using `K` fixed, distinct
reference seeds and then average their coefficient matrices in the linear
model or their predictions in the ReLU model.

## Inference

The repository uses one explicit covariance convention. If `IF` is the
`n x q` matrix of observation-level influence values, then

```text
V = IF.T @ IF / n
```

estimates the asymptotic covariance of `sqrt(n) * (theta_hat - theta_0)`.
Consequently, the estimated covariance of `theta_hat` is `V / n`, and its
marginal standard errors are `sqrt(diag(V) / n)`.

### Influence-function estimators

OLS, coordinatewise Huber, and radial Huber use their fitted estimating
equations. The generic implicit calculation is

```text
IF_i = -A^{-1} C_i,
```

where `A` is the parameter-score derivative and `C_i` is the derivative along
the empirical contamination path `delta_i - F_n`.

CW-ASM uses `fit_coordinatewise_asm_with_inference`, which fits every response
coordinate once and retains the R package outputs `Cov_mat`, `std_errs`,
`zvals`, `sig_vals`, `psi`, and `I_mat`. `Cov_mat` is the package-authoritative
marginal covariance of `sqrt(n) * (theta_hat - theta_0)`. The optional
`asm_inference_from_fit_state` adapter uses simultaneous observation-level
scores to estimate cross-response covariance blocks. These empirical joint
blocks are reported separately because the R package does not estimate them.

MCQR currently has no validated IF implementation.

CW-CQR is a separate coordinatewise composite-quantile baseline. Its public
fit API is `otrank.estimators.fit_cw_cqr`; the packaged R bridge calls
`cqrReg::cqr.fit` with a common slope and quantile-specific nuisance
intercepts. The formal setting uses 20 levels `tau_l=l/21`, `method="ip"`,
`maxit=1000`, and tolerance `1e-7`. `cw_cqr_covariance` profiles the 20
intercepts through the full bread for each response, arranges the
observation-level slope influence functions using the standard column-major
`vec(B)`, and forms the cross-response sandwich covariance.

The primary OTRR inference API is `plugin_covariance`. It uses the plug-in
Hessian from the asymptotic-normality theorem:

```text
X_c = X - mean(X)
T_c = T - mean(T)
H_plugin = kron(X_c.T @ X_c / n, mean OT-map Jacobian)
```

with the fitted-rank observation-level score

```math
\widehat\psi_i
=
\mathrm{vec}\{(\widehat T(\widehat\xi_i)-\bar T)(X_i-\bar X)^\top\},
\qquad
\widehat{\mathrm{IF}}_i
=
\widehat H^{-1}(\widehat\psi_i-\bar\psi).
```

The fitted residual/intercept convention is preserved separately from score
centering. Schema-v6 direct-solve artifacts record both centering conventions,
so uncentered artifacts cannot be silently resumed as centered ones. This is
the rank-score plug-in approximation; it does not substitute the full Hessian
of the fixed-temperature raw objective. Archived IF reconstructions have a
separate artifact kind and retain the original dual diagnostics without
claiming new dual solves.

### Case-weight influence covariance

The optional `case_weight_if_covariance` differentiates the finite-sample
dense-PNW objective with respect to observation weights. For normalized case
weights $w=(w_1,\ldots,w_n)$, define

```math
\widehat L_{n,k}(\theta,w)
=
\sum_{i=1}^n
w_i
\left\langle
R_i(\theta),
\widehat T_{\theta,w}^{(k)}\{R_i(\theta)\}
\right\rangle,
```

where $\widehat T_{\theta,w}^{(k)}$ depends on the weighted empirical
residual distribution.

Let

```math
\widehat A_k
=
\nabla_\theta^2
\widehat L_{n,k}(\widehat\theta_k,w_0),
\qquad
w_0=n^{-1}\mathbf 1_n,
```

and consider the contamination path

```math
w^{(i)}(t)=(1-t)w_0+t e_i.
```

The case-weight score derivative and influence function are

```math
\widehat C_{ik}
=
\left.
\frac{d}{dt}
\nabla_\theta
\widehat L_{n,k}\{\widehat\theta_k,w^{(i)}(t)\}
\right|_{t=0},
\qquad
\widehat{\mathrm{IF}}_{ik}^{\mathrm{cw}}
=
-\widehat A_k^{-1}\widehat C_{ik}.
```

For a $K$-reference ensemble, the influence functions are averaged before
forming the covariance:

```math
\widehat{\mathrm{IF}}_i^{\mathrm{cw,ens}}
=
\frac1K\sum_{k=1}^K
\widehat{\mathrm{IF}}_{ik}^{\mathrm{cw}},
\qquad
\widehat V_{\mathrm{cw}}
=
\frac1n\sum_{i=1}^n
\widehat{\mathrm{IF}}_i^{\mathrm{cw,ens}}
\widehat{\mathrm{IF}}_i^{\mathrm{cw,ens}\top}.
```

The required higher-order derivatives use dense PyTorch even when fitting used dense Triton, since the custom Triton backward does not support double backward. For `intercept_mode="pilot"`, the saved `x_mean` and `y_mean` are treated as fixed tuning quantities.

## License

Copyright (C) 2026 Yizhe Ding.

This project is licensed under the GNU General Public License, version 3 only
(`GPL-3.0-only`). You may use, modify, and redistribute the code subject to the
terms of that license. See [`LICENSE`](LICENSE) for the complete license text.

Third-party software and datasets used by the reproduction workflows remain
subject to their respective licenses and terms of use.
