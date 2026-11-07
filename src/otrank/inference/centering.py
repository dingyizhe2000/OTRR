"""Numerically checked centering of archived common-design plug-in ensembles.

This migration does not solve OT or differentiate the finite-temperature loss.
It relies on the common design Kronecker factor in the archived plug-in breads.
"""

from __future__ import annotations

import numpy as np


def recenter_plugin_ensemble_influence(
    x: np.ndarray,
    influence: np.ndarray,
    *,
    factorization_tolerance: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float | str | bool]]:
    """Recover centered slope IFs without re-solving the reference OT maps.

    For M=mean(XX'), let v_i=M^{-1}X_i. Ignoring a numerically tiny ridge,
    each archived d-by-p IF has form F_i=w_i v_i'-mean(w v'), where w_i
    averages reference-specific inverse-Jacobian rank scores. The missing
    mean is identified by projecting F_i' perpendicular to v_i. Reconstruction
    error is checked explicitly; penalized/non-Kronecker inputs are not assumed
    compatible. Columns must follow column-major vec(B), with no intercept
    column in X. Platform reproduction need only meet the stated tolerance.
    """
    design = np.asarray(x, dtype=np.float64)
    phi = np.asarray(influence, dtype=np.float64)
    if design.ndim != 2 or phi.ndim != 2 or len(design) != len(phi):
        raise ValueError("x and influence must be matrices with matching observations")
    n, p = design.shape
    if p < 2 or n <= p or phi.shape[1] == 0 or phi.shape[1] % p:
        raise ValueError("reconstruction requires n>p>=2 and a slope IF with d*p columns")
    if not np.isfinite(design).all() or not np.isfinite(phi).all():
        raise ValueError("x and influence must be finite")
    if not 0 < factorization_tolerance < 1:
        raise ValueError("factorization_tolerance must be between zero and one")
    d = phi.shape[1] // p
    raw_moment = design.T @ design / n
    centered_design = design - design.mean(axis=0)
    centered_moment = centered_design.T @ centered_design / n
    for moment in (raw_moment, centered_moment):
        if np.linalg.matrix_rank(moment) != p:
            raise ValueError("design moments must be full rank; exclude an intercept column")
        np.linalg.cholesky(moment)
    v = np.linalg.solve(raw_moment, design.T).T
    norms = np.einsum("np,np->n", v, v)
    if np.any(norms <= 0):
        raise ValueError("cannot recover outer-product scores for a zero design row")
    source_mean = phi.mean(axis=0)
    scores = (phi-source_mean).reshape(n, p, d)
    projection_sum = n*np.eye(p) - v.T @ (v/norms[:, None])
    inner = np.einsum("np,npd->nd", v, scores)
    rhs = -scores.sum(axis=0) + np.einsum("np,nd->pd", v/norms[:, None], inner)
    missing_mean = np.linalg.solve(projection_sum, rhs)
    w = np.einsum("np,npd->nd", v, scores+missing_mean[None]) / norms[:, None]
    reconstructed = np.einsum("np,nd->npd", v, w) - missing_mean[None]
    denominator = np.linalg.norm(scores)
    if denominator == 0:
        raise ValueError("an identically zero influence cannot identify whitening scores")
    residual = float(np.linalg.norm(scores-reconstructed) / denominator)
    if not np.isfinite(residual) or residual > factorization_tolerance:
        raise ValueError(
            f"common-design IF reconstruction residual {residual:.3e} exceeds "
            f"{factorization_tolerance:.3e}; check data, ordering and bread structure"
        )
    vc = np.linalg.solve(centered_moment, centered_design.T).T
    result = np.einsum("np,nd->npd", vc, w-w.mean(axis=0)).reshape(n, p*d)
    result -= result.mean(axis=0)
    return result, {
        "construction": "common_design_kronecker_reconstruction",
        "design_centering": "sample_mean",
        "rank_centering": "sample_mean",
        "factorization_relative_error": residual,
        "factorization_tolerance": factorization_tolerance,
        "original_if_mean_norm": float(np.linalg.norm(source_mean)),
        "design_condition_number": float(np.linalg.cond(centered_moment)),
        "numerically_approximate": True,
        "maps_recomputed": False,
    }
