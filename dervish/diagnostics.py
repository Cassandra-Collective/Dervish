"""Measurements the claims rest on. All functions are torch.no_grad and cheap
at lab scale (matrices of a few hundred rows)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .matricize import FrameView, matricize
from .polar import orthonormality_error


@torch.no_grad()
def singular_values(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(matrix.to(torch.float32))


@torch.no_grad()
def spectral_norm(matrix: torch.Tensor) -> float:
    return float(singular_values(matrix).max())


@torch.no_grad()
def stable_rank(matrix: torch.Tensor) -> float:
    """||W||_F^2 / ||W||_2^2 : how many singular directions carry the energy."""
    s = singular_values(matrix)
    top = float(s.max())
    if top <= 0.0:
        return 0.0
    return float((s * s).sum()) / (top * top)


@torch.no_grad()
def effective_rank(matrix: torch.Tensor) -> float:
    """exp(entropy of the normalised singular values) (Roy and Vetterli, 2007)."""
    s = singular_values(matrix)
    total = float(s.sum())
    if total <= 0.0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def matrix_report(matrix: torch.Tensor) -> dict[str, float]:
    s = singular_values(matrix)
    top = float(s.max())
    frob_sq = float((s * s).sum())
    return {
        "spectral_norm": top,
        "frobenius_norm": math.sqrt(frob_sq),
        "stable_rank": frob_sq / (top * top) if top > 0 else 0.0,
        "effective_rank": effective_rank(matrix),
        "sigma_min": float(s.min()),
    }


@torch.no_grad()
def hidden_matrix_report(
    model: nn.Module,
    is_hidden,
    views: dict[str, FrameView] | None = None,
) -> dict[str, Any]:
    """Aggregate spectral statistics over the model's hidden matrices.

    `is_hidden(name, parameter)` selects the parameters (the same predicate the
    optimizer builders use). A parameter with a FrameView is reported per frame
    and aggregated with max/mean, so a per-head view reports per-head
    spectral norms.
    """
    per_param: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not is_hidden(name, parameter):
            continue
        view = (views or {}).get(name, FrameView.auto())
        wide = matricize(parameter.detach(), view).wide.to(torch.float32)
        flat = wide.reshape(-1, wide.size(-2), wide.size(-1))
        reports = [matrix_report(f) for f in flat]
        per_param.append(
            {
                "name": name,
                "frames": len(reports),
                "spectral_norm_max": max(r["spectral_norm"] for r in reports),
                "stable_rank_mean": sum(r["stable_rank"] for r in reports) / len(reports),
                "effective_rank_mean": sum(r["effective_rank"] for r in reports) / len(reports),
                "sigma_min_min": min(r["sigma_min"] for r in reports),
                "orthonormality_error_max": float(
                    orthonormality_error(flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)).max()
                ),
            }
        )
    if not per_param:
        return {"hidden_tensors": 0}
    return {
        "hidden_tensors": len(per_param),
        "hidden_spectral_norm_max": max(p["spectral_norm_max"] for p in per_param),
        "hidden_spectral_norm_mean": sum(p["spectral_norm_max"] for p in per_param) / len(per_param),
        "hidden_stable_rank_mean": sum(p["stable_rank_mean"] for p in per_param) / len(per_param),
        "hidden_effective_rank_mean": sum(p["effective_rank_mean"] for p in per_param) / len(per_param),
        "hidden_sigma_min_min": min(p["sigma_min_min"] for p in per_param),
        "hidden_row_orthonormality_error_max": max(p["orthonormality_error_max"] for p in per_param),
        "per_parameter": per_param,
    }
