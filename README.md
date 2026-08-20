# Orthogonal Invariance in Multivariate-Response Regression via Optimal Transport Rank Regression

This repository contains the public reproducibility code for the paper
"Orthogonal Invariance in Multivariate-Response Regression via Optimal
Transport Rank Regression." It includes the optimal transport rank regression
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
RTX 3090 24 GB GPU. Additional linear-simulation jobs used NVIDIA A100 PCIe
40 GB GPUs.

The ASM and CW-CQR baselines use the R packages `asm` and `cqrReg`. Install R
and then run:

```bash
Rscript -e 'install.packages(c("asm", "cqrReg"), repos="https://cloud.r-project.org")'
```

If `cqrReg` is installed in a nonstandard user library, set `CQR_R_LIB` to
that directory. Otherwise, the bridge uses R's normal `.libPaths()`, including
`R_LIBS_USER`.

## Reproduce the Paper Results

Run the commands below from the repository root. The large fitted-state and
inference bundles are not tracked in Git; place them under `results/` before
running the paper-output builders. To regenerate all empirical tables and
figures from those bundles in one step, use
`PYTHONPATH=src python reproduce_paper_outputs.py --project-root .`.

### Linear simulation

`configs/simulation_linear.json` is the configuration for the 50-repetition
linear experiment. The first command reruns the estimators from scratch. The
second is an independent workflow that rebuilds the canonical bundle, Tables
3--4, and Figures S2.1--S2.2 from the retained results.

```bash
python run_simulation_study.py --config configs/simulation_linear.json \
  --out-root reproduced_results/linear
PYTHONPATH=src python -m otrank.results_analysis.finalize_linear_simulation \
  --project-root . --output-dir results/simulation/linear_n1000_p20_d10_final
```

### ReLU simulation

`configs/simulation_relu.json` contains three execution stages for the
50-repetition ReLU experiment: classical baselines, CW-CQR, and
CW-CQR-initialized OTRR. The first command reruns all stages. The second is an
independent workflow that rebuilds the canonical ReLU bundle and Table S2.1
from the retained results.

```bash
python run_simulation_study.py --config configs/simulation_relu.json \
  --out-root reproduced_results/relu
PYTHONPATH=src python -m otrank.results_analysis.finalize_relu_simulation \
  --project-root . --output-dir results/simulation/relu_n1000_p8_d4_final
```

### NHANES real-data application

Place the retained fitted states and inference artifacts under
`results/real_data/nhanes_p43_final/source/`. The following command validates
these artifacts and rebuilds the cross-validation summaries, Table 5, and the
associated supplementary outputs. It does not refit the estimators.

```bash
python run_real_data_experiment.py --config configs/main_real_data.json \
  --result-root results/real_data/nhanes_p43_final
```

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
H_plugin = kron(design second moment, mean OT-map Jacobian)
```

with the fitted-rank observation-level score

```math
\widehat\psi_i
=
\operatorname{vec}\{\widehat T(\widehat\xi_i)X_i^\top\},
\qquad
\widehat{\operatorname{IF}}_i
=
\widehat H^{-1}(\widehat\psi_i-\bar\psi).
```

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
\widehat{\operatorname{IF}}_{ik}^{\rm cw}
=
-\widehat A_k^{-1}\widehat C_{ik}.
```

For a $K$-reference ensemble, the influence functions are averaged before
forming the covariance:

```math
\widehat{\operatorname{IF}}_i^{\rm cw,ens}
=
\frac1K\sum_{k=1}^K
\widehat{\operatorname{IF}}_{ik}^{\rm cw},
\qquad
\widehat V_{\rm cw}
=
\frac1n\sum_{i=1}^n
\widehat{\operatorname{IF}}_i^{\rm cw,ens}
\widehat{\operatorname{IF}}_i^{\rm cw,ens\top}.
```

The required higher-order derivatives use dense PyTorch even when fitting used dense Triton, since the custom Triton backward does not support double backward. For `intercept_mode="pilot"`, the saved `x_mean` and `y_mean` are treated as fixed tuning quantities.

## License

Copyright (C) 2026 Yizhe Ding.

This project is licensed under the GNU General Public License, version 3 only
(`GPL-3.0-only`). You may use, modify, and redistribute the code subject to the
terms of that license. See [`LICENSE`](LICENSE) for the complete license text.

Third-party software and datasets used by the reproduction workflows remain
subject to their respective licenses and terms of use.
