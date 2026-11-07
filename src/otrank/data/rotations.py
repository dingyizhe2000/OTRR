"""Seed-only response-rotation replay with explicit numerical-environment checks."""

from __future__ import annotations

import torch

from .noise import make_generator, random_orthogonal

ALGORITHM = "torch_randn_square_qr_sign_corrected_v1"


def rotation_spec(seed: int, dimension: int, device: torch.device, dtype: torch.dtype) -> dict:
    if device.type not in {"cpu", "cuda"} or dtype not in {torch.float32, torch.float64}:
        raise ValueError("paper rotation replay supports explicit CPU/CUDA and float32/float64")
    return {"algorithm": ALGORITHM, "seed": int(seed), "dimension": int(dimension),
            "rng_device": str(device), "dtype": str(dtype).removeprefix("torch."),
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda if device.type == "cuda" else None}


def replay_rotation(spec: dict, *, output_device: torch.device | str | None = None) -> torch.Tensor:
    required = {"algorithm", "seed", "dimension", "rng_device", "dtype", "torch_version", "cuda_runtime"}
    if not required.issubset(spec) or spec["algorithm"] != ALGORITHM:
        raise ValueError("rotation replay needs its recorded generator, seed, device, dtype and runtime metadata")
    if spec["torch_version"] != torch.__version__:
        raise RuntimeError(f"Rotation replay requires recorded PyTorch {spec['torch_version']}; current version is {torch.__version__}.")
    device = torch.device(spec["rng_device"])
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("unsupported recorded rotation RNG device")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Recorded CUDA rotation cannot be regenerated on CPU; use its matching CUDA environment or retained paper metrics.")
        if spec["cuda_runtime"] != torch.version.cuda:
            raise RuntimeError("CUDA rotation replay requires the recorded CUDA runtime")
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(spec["dtype"])
    if dtype is None or int(spec["dimension"]) < 1:
        raise ValueError("invalid recorded rotation dimension or dtype")
    q = random_orthogonal(int(spec["dimension"]), make_generator(int(spec["seed"]), device=device), device=device, dtype=dtype)
    return q.to(output_device) if output_device is not None else q
