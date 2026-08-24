from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import torch

import otrank
from otrank import LinearOTRankConfig, fit_ot_rank_linear


ROOT = Path(__file__).resolve().parents[1]


def test_import_without_triton_in_subprocess() -> None:
    script = """
import sys
import torch
sys.modules['triton'] = None
sys.modules['triton.language'] = None
import otrank.ot_maps.pnw_triton as module
try:
    module.triton_pnw_row_loss(torch.zeros(2, 2), torch.zeros(3, 2), torch.zeros(3), epsilon=0.1, cost_scale=1.0)
except ImportError as error:
    assert 'Triton is required' in str(error)
else:
    raise AssertionError('missing Triton call did not fail')
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )


def test_package_modules_import() -> None:
    failures = []
    for module in pkgutil.walk_packages(otrank.__path__, prefix="otrank."):
        try:
            importlib.import_module(module.name)
        except Exception as error:  # pragma: no cover - assertion reports module
            failures.append((module.name, repr(error)))
    assert failures == []


def _tiny_fit(tmp_path: Path, backend: str):
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(12, 2, generator=generator)
    coefficient = torch.tensor([[0.5, -0.25], [0.2, 0.4]])
    y = x @ coefficient.T + 0.05 * torch.randn(12, 2, generator=generator)
    config = LinearOTRankConfig(
        m_ref=8,
        epsilon=0.2,
        sinkhorn_iters=2,
        steps=1,
        lr=1e-3,
        trace_every=1,
        checkpoint_every=1,
        backend=backend,
        device="cpu",
    )
    return fit_ot_rank_linear(x, y, config=config, save_dir=tmp_path / backend)


def test_dense_pytorch_tiny_cpu_fit(tmp_path: Path) -> None:
    fit = _tiny_fit(tmp_path, "dense_pytorch")
    assert fit.coef.shape == (2, 2)
    assert torch.isfinite(fit.coef).all()


def test_default_dense_triton_config_uses_documented_cpu_fallback(tmp_path: Path) -> None:
    fit = _tiny_fit(tmp_path, "dense_triton")
    assert fit.coef.shape == (2, 2)
    assert torch.isfinite(fit.coef).all()
