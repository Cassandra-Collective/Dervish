"""Nearest unitary to a drifted gate, with the optimizer's own polar routine.

A CNOT is perturbed by a random complex matrix whose Frobenius norm is drawn from
an interval (an interval-valued drift). The nearest
unitary in Frobenius norm is the polar factor of the drifted matrix. `dervish.polar`
works on real matrices, so the complex 4x4 gate is realified to an 8x8 real matrix;
the realification respects products and conjugate-transposes, so the polar factor
of the real matrix is the realification of the complex polar factor.

Prints, over a few drift intervals, the Frobenius distance of the drifted gate and of
its nearest unitary to the ideal CNOT, and the process fidelity
F = |Tr(U_ideal^H U)|^2 / d^2 of the nearest unitary (fidelity is only meaningful for a
unitary, which is why it is not printed for the drifted matrix).
Deterministic (seeded).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dervish.polar import orthonormality_error, polar  # noqa: E402


def realify(a: torch.Tensor) -> torch.Tensor:
    re, im = a.real, a.imag
    return torch.cat([torch.cat([re, -im], dim=1), torch.cat([im, re], dim=1)], dim=0)


def unrealify(r: torch.Tensor) -> torch.Tensor:
    d = r.size(0) // 2
    return torch.complex(r[:d, :d], r[d:, :d])


def process_fidelity(u_ideal: torch.Tensor, u: torch.Tensor) -> float:
    d = u.size(0)
    tr = torch.trace(u_ideal.conj().T @ u)
    return float((tr.abs() ** 2) / (d * d))


def main() -> None:
    torch.manual_seed(20261009)
    cnot = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.complex64
    )
    print("drift interval  sample  ||E||_F  ||drifted-CNOT||_F  ||nearest-CNOT||_F  F(nearest unitary)  unitarity err")
    for low, high in [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.60)]:
        for sample in range(3):
            noise = torch.randn(4, 4, dtype=torch.complex64)
            norm = low + (high - low) * torch.rand(()).item()
            noise = noise / torch.linalg.matrix_norm(noise) * norm
            drifted = cnot + noise
            nearest = unrealify(polar(realify(drifted)))
            err = float(orthonormality_error(realify(nearest)))
            d_drift = float(torch.linalg.matrix_norm(drifted - cnot))
            d_near = float(torch.linalg.matrix_norm(nearest - cnot))
            print(
                f"[{low:.2f}, {high:.2f}]    {sample}     {norm:6.3f}        {d_drift:8.5f}            {d_near:8.5f}          "
                f"{process_fidelity(cnot, nearest):8.5f}          {err:9.2e}"
            )
    print()
    print("The nearest unitary is the polar factor of the drifted gate, computed by the same")
    print("Newton-Schulz iteration Dervish uses to rotate weight frames (dervish/polar.py).")


if __name__ == "__main__":
    main()
