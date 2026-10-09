import torch

from dervish.polar import (
    orthonormality_error,
    polar,
    polar_cubic,
    polar_exact,
    tangent_project,
    zeropower_quintic,
)


def _shapes():
    return [(8, 32), (32, 8), (64, 64), (3, 16, 40), (3, 40, 16), (96, 24)]


def test_polar_matches_svd_reference_on_gaussian_matrices():
    g = torch.Generator().manual_seed(7)
    for shape in _shapes():
        w = torch.randn(*shape, generator=g) * 0.02
        q = polar(w)
        ref = polar_exact(w)
        assert torch.allclose(q, ref, atol=2e-4), shape
        assert float(orthonormality_error(q).max()) < 1e-5, shape


def test_polar_handles_badly_conditioned_matrices():
    g = torch.Generator().manual_seed(11)
    u, _ = torch.linalg.qr(torch.randn(32, 32, generator=g))
    v, _ = torch.linalg.qr(torch.randn(48, 48, generator=g))
    sigma = torch.logspace(0, -2, 32)  # condition number 100
    w = u @ torch.diag(sigma) @ v[:32, :]
    q = polar(w)
    assert float(orthonormality_error(q).max()) < 1e-4
    assert torch.allclose(q, polar_exact(w), atol=1e-3)


def test_quintic_alone_is_not_a_polar_factor():
    # Muon's iteration spreads singular values in roughly [0.68, 1.13]; the
    # retraction cannot use it alone. This pins the reason the cubic exists.
    g = torch.Generator().manual_seed(3)
    w = torch.randn(64, 256, generator=g)
    x = zeropower_quintic(w, steps=5)
    s = torch.linalg.svdvals(x)
    assert float(s.min()) > 0.5 and float(s.max()) < 1.3
    assert float(orthonormality_error(x).max()) > 1e-2


def test_cubic_retraction_is_exact_for_small_tangent_steps():
    g = torch.Generator().manual_seed(5)
    q = polar(torch.randn(48, 128, generator=g))
    d = torch.randn(48, 128, generator=g)
    t = tangent_project(d, q)
    # tangent means T Q^T is skew-symmetric
    a = t @ q.mT
    assert torch.allclose(a, -a.mT, atol=1e-5)
    for lr in (0.01, 0.05, 0.2):
        step = q - lr * t / torch.linalg.matrix_norm(t, ord=2)
        s = torch.linalg.svdvals(step)
        assert float(s.min()) >= 1.0 - 1e-5 and float(s.max()) <= (1.0 + lr * lr) ** 0.5 + 1e-5
        q_new = polar_cubic(step, steps=2)
        assert float(orthonormality_error(q_new)) < 1e-5, lr


def test_batched_and_tall_inputs_round_trip():
    g = torch.Generator().manual_seed(9)
    w = torch.randn(4, 50, 20, generator=g)
    q = polar(w)
    assert q.shape == w.shape
    assert float(orthonormality_error(q).max()) < 1e-5
