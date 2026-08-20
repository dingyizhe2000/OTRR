"""Interfaces for OT map estimators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


class FittedOTMap(Protocol):
    """Callable fitted transport map from residuals to reference scores."""

    def transform(self, residuals: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        """Evaluate the fitted OT map at residual vectors."""

    def potential(self, residuals: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        """Evaluate the fitted convex potential, when available."""

    def state_dict(self) -> dict[str, Any]:
        """Return warm-start state for the next OT-map iteration."""


class OTMapEstimator(Protocol):
    """Estimator that fits an OT map from residual samples."""

    def fit(
        self,
        residuals: torch.Tensor,
        init_state: dict[str, Any] | None = None,
        save_dir: str | Path | None = None,
    ) -> FittedOTMap:
        """Fit an OT map using residual samples."""


@dataclass
class UnavailableOTMapEstimator:
    """Placeholder used until a concrete OT map estimator is provided."""

    message: str = (
        "OT map estimation is intentionally left unimplemented. "
        "Plug in the project's neural or empirical OT estimator by implementing "
        "OTMapEstimator.fit(residuals) -> FittedOTMap."
    )

    def fit(
        self,
        residuals: torch.Tensor,
        init_state: dict[str, Any] | None = None,
        save_dir: str | Path | None = None,
    ) -> FittedOTMap:
        raise NotImplementedError(self.message)
