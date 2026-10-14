"""MONA (arXiv:2605.26842) as one change to the repo's Muon, and inside Dervish."""

import pytest
import torch
from torch import nn

from dervish import build_dervish, build_muon
from dervish.optimizer import mona_accelerate, mona_tied_alpha
from dervish.polar import orthonormality_error


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(11, 16)
        self.blocks = nn.ModuleDict({"a": nn.Linear(16, 24, bias=False), "b": nn.Linear(24, 16, bias=False)})
        self.head = nn.Linear(16, 11)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        x = x + self.blocks["b"](torch.tanh(self.blocks["a"](x)))
        return self.head(x)


def _run(optimizer, model: nn.Module, steps: int = 12, seed: int = 3) -> list[float]:
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        x = torch.randint(0, 11, (4, 6), generator=g)
        y = torch.randint(0, 11, (4, 6), generator=g)
        loss = nn.functional.cross_entropy(model(x).view(-1, 11), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def _muon(model: nn.Module, **muon_kwargs):
    optimizer, _ = build_muon(model.named_parameters(), muon_kwargs={"lr": 0.02, **muon_kwargs}, adam_kwargs={"lr": 1e-3})
    return optimizer


def test_tied_alpha_matches_the_paper() -> None:
    # section 8 of the paper: beta_a 0.99, 0.98, 0.975 -> alpha -50, -25, -20
    assert mona_tied_alpha(0.99) == pytest.approx(-50.0)
    assert mona_tied_alpha(0.98) == pytest.approx(-25.0)
    assert mona_tied_alpha(0.975) == pytest.approx(-20.0)
    with pytest.raises(ValueError):
        mona_tied_alpha(1.0)


def test_alpha_zero_is_plain_muon_bit_for_bit() -> None:
    torch.manual_seed(0)
    a = Tiny()
    b = Tiny()
    b.load_state_dict(a.state_dict())
    losses_a = _run(_muon(a), a)
    losses_b = _run(_muon(b, mona_beta=0.9, mona_alpha=0.0), b)
    assert losses_a == losses_b
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_mona_changes_the_trajectory() -> None:
    torch.manual_seed(0)
    a = Tiny()
    b = Tiny()
    b.load_state_dict(a.state_dict())
    _run(_muon(a), a)
    _run(_muon(b, mona_beta=0.9), b)
    assert not torch.equal(a.blocks["a"].weight, b.blocks["a"].weight)


def test_tied_term_is_the_mean_of_gradient_and_its_slow_ema() -> None:
    # The identity behind the audit note in docs: with alpha tied to beta,
    # G~_k = (G_k + E_{k-1}) / 2, E the zero-initialised beta-EMA of G_1..G_{k-1}.
    beta = 0.9
    g = torch.Generator().manual_seed(1)
    grads = [torch.randn(5, 7, generator=g, dtype=torch.float64) for _ in range(30)]
    state: dict = {}
    ema = torch.zeros(5, 7, dtype=torch.float64)
    for grad in grads:
        accelerated = mona_accelerate(grad, state, beta, None)
        assert torch.allclose(accelerated, 0.5 * (grad + ema), atol=1e-12)
        ema = beta * ema + (1.0 - beta) * grad


def test_dervish_with_mona_keeps_the_manifold_and_the_cap() -> None:
    torch.manual_seed(0)
    model = Tiny()
    optimizer, _ = build_dervish(
        model.named_parameters(),
        frame_kwargs={"lr": 0.05, "scale_lr": 0.05, "scale_cap": 0.5, "mona_beta": 0.9},
        adam_kwargs={"lr": 1e-3},
    )
    optimizer.initialize()
    _run(optimizer, model, steps=20)
    for parameter in optimizer.frame_parameters():
        state = optimizer.state[parameter]
        assert "mona_accel" in state
        assert float(orthonormality_error(state["frame"]).max()) < 1e-4
        assert float(torch.linalg.matrix_norm(parameter.detach(), ord=2)) <= 0.5 + 1e-4
