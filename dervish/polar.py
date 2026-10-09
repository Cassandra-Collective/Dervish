"""Polar factors by Newton-Schulz iteration.

Two iterations live here because they do two different jobs.

`zeropower_quintic` is the Muon iteration (coefficients 3.4445, -4.7750,
2.0315, as used in Keller Jordan's Muon and copied verbatim into many
Muon trainers). Five steps push every singular value of a Frobenius-normalised matrix
into roughly [0.68, 1.13]. That is what an update direction needs: every
direction alive, none dominant. It is not what a manifold retraction may use,
because the result is not orthonormal to working precision.

`polar_cubic` is the classical Newton-Schulz iteration
X <- 1.5 X - 0.5 (X X^T) X. It converges quadratically to the polar factor
whenever every singular value lies in (0, sqrt(3)). Dervish feeds it matrices
whose singular values sit in [1, sqrt(1 + lr^2)], so two steps reach fp32
precision.

`polar` composes the two for a general matrix: quintic steps to compress the
spectrum, cubic steps to finish. `polar_exact` is the SVD reference used by the
tests.

All functions accept (..., m, n) tensors, batched or not, and treat tall
matrices by transposition, so "orthonormal rows" below means "orthonormal rows
when m <= n, orthonormal columns otherwise".
"""

from __future__ import annotations

import torch

QUINTIC_A = 3.4445
QUINTIC_B = -4.7750
QUINTIC_C = 2.0315


def _as_wide(matrix: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if matrix.ndim < 2:
        raise ValueError("polar iterations expect a matrix or batched matrices")
    if matrix.size(-2) <= matrix.size(-1):
        return matrix, False
    return matrix.mT, True


def _from_wide(matrix: torch.Tensor, transposed: bool) -> torch.Tensor:
    return matrix.mT if transposed else matrix


def zeropower_quintic(
    matrix: torch.Tensor,
    steps: int = 5,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Muon's orthogonalising iteration. Returns a tensor in `dtype`.

    GPU trainers usually run this in bfloat16 on CUDA. On CPU bfloat16 matmul
    is slow and no more accurate, so the lab passes float32 there; the
    coefficients and step count are unchanged.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    x, transposed = _as_wide(matrix)
    x = x.to(dtype)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        a = x @ x.mT
        b = QUINTIC_B * a + QUINTIC_C * (a @ a)
        x = QUINTIC_A * x + b @ x
    return _from_wide(x, transposed)


def polar_cubic(matrix: torch.Tensor, steps: int = 2) -> torch.Tensor:
    """Classical Newton-Schulz polar iteration in the input dtype.

    Converges to the nearest orthonormal matrix when the singular values of the
    input lie in (0, sqrt(3)). Quadratic: an error of 1e-2 becomes about 1e-4,
    then about 1e-8.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    x, transposed = _as_wide(matrix)
    for _ in range(steps):
        x = 1.5 * x - 0.5 * ((x @ x.mT) @ x)
    return _from_wide(x, transposed)


def polar(
    matrix: torch.Tensor,
    tol: float = 1e-6,
    max_quintic_steps: int = 12,
    max_cubic_steps: int = 30,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Polar factor (nearest orthonormal matrix) of a general matrix, float32.

    Adaptive: quintic steps run until the spectrum is compressed (orthonormality
    error below 0.25, where the cubic iteration converges fast), then cubic
    steps run until the error is below `tol` or the step budget is spent. For
    Gaussian-initialised weights, square ones included, the error after this
    routine is below 1e-5 in float32 (tests/test_polar.py measures it). A
    rank-deficient input cannot reach the tolerance; the routine then returns
    the best iterate rather than raising.
    """
    x, transposed = _as_wide(matrix)
    x = x.to(torch.float32)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(max_quintic_steps):
        a = x @ x.mT
        b = QUINTIC_B * a + QUINTIC_C * (a @ a)
        x = QUINTIC_A * x + b @ x
        if float(orthonormality_error(x).max()) < 0.25:
            break
    for _ in range(max_cubic_steps):
        x = 1.5 * x - 0.5 * ((x @ x.mT) @ x)
        if float(orthonormality_error(x).max()) < tol:
            break
    return _from_wide(x, transposed)


def polar_exact(matrix: torch.Tensor) -> torch.Tensor:
    """SVD polar factor U V^T, computed in float64.

    Used by the tests as the reference and by Dervish once per parameter at
    initialisation, where exactness matters more than speed (an ill-conditioned
    square initial weight is the hard case for the iteration above)."""
    u, _s, vh = torch.linalg.svd(matrix.to(torch.float64), full_matrices=False)
    return (u @ vh).to(matrix.dtype)


def orthonormality_error(matrix: torch.Tensor) -> torch.Tensor:
    """||Q Q^T - I||_F for wide Q (||Q^T Q - I||_F for tall Q), per batch element."""
    x, _transposed = _as_wide(matrix)
    gram = x @ x.mT
    eye = torch.eye(gram.size(-1), dtype=gram.dtype, device=gram.device)
    return (gram - eye).norm(dim=(-2, -1))


def tangent_project(direction: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Project `direction` onto the tangent space of the Stiefel manifold at `frame`.

    Both are wide (..., m, n) with frame Q satisfying Q Q^T = I. The tangent
    space is {D : D Q^T + Q D^T = 0}; the projection removes the symmetric part
    of D Q^T:  P(D) = D - sym(D Q^T) Q.
    """
    a = direction @ frame.mT
    sym = 0.5 * (a + a.mT)
    return direction - sym @ frame
