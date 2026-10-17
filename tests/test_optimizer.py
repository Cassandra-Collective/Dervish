import math

import pytest
import torch
from torch import nn

from dervish import Dervish, FrameView, MuonWithAuxAdam, build_dervish, build_muon
from dervish.diagnostics import hidden_matrix_report, spectral_norm, stable_rank
from dervish.matricize import matricize
from dervish.polar import orthonormality_error


class Toy(nn.Module):
    """A fake 'blocks.' naming so the default frame predicate selects hidden matrices."""

    def __init__(self, d: int = 32, hidden: int = 96, n_head: int = 4, experts: int = 3) -> None:
        super().__init__()
        self.embed = nn.Embedding(17, d)
        self.blocks = nn.ModuleDict(
            {
                "attn": nn.Linear(d, 3 * d, bias=True),  # fused qkv, viewed per head
                "up": nn.Linear(d, hidden, bias=False),  # tall
                "down": nn.Linear(hidden, d, bias=False),  # wide
                "conv": nn.Conv2d(d, d, kernel_size=3, padding=1, bias=False),  # 4D
                "proj": nn.Linear(d, d, bias=False),  # square
            }
        )
        self.experts = nn.Parameter(torch.randn(experts, d, hidden) * 0.05)  # (E, m, n)
        self.head = nn.Linear(d, 17)
        self.n_head = n_head
        self.d = d

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)  # (B, T, d)
        q, k, v = self.blocks["attn"](x).chunk(3, dim=-1)
        x = x + torch.tanh(q + k + v)
        x = x + self.blocks["down"](torch.tanh(self.blocks["up"](x)))
        img = x.transpose(1, 2).unsqueeze(-1)  # (B, d, T, 1)
        x = x + self.blocks["conv"](img).squeeze(-1).transpose(1, 2)
        x = x + torch.tanh(self.blocks["proj"](x))
        gate = torch.softmax(x.mean(dim=1) @ torch.ones(self.d, self.experts.size(0)), dim=-1)  # (B, E)
        mixed = torch.einsum("be,emn->bmn", gate, self.experts)  # (B, d, hidden)
        x = x + torch.tanh(torch.einsum("btm,bmn->btn", x, mixed)).mean(dim=-1, keepdim=True)
        return self.head(x)


def _frame_predicate(name: str, p: nn.Parameter) -> bool:
    return p.ndim >= 2 and (name.startswith("blocks.") or name == "experts")


def _views(model: Toy) -> dict[str, FrameView]:
    return {"blocks.attn.weight": FrameView.heads(3 * model.n_head, model.d // model.n_head, model.d)}


def _train(model: nn.Module, optimizer, steps: int, seed: int, fixed_batch: bool = False) -> list[float]:
    g = torch.Generator().manual_seed(seed)
    losses = []
    fixed = (torch.randint(0, 17, (4, 8), generator=g), torch.randint(0, 17, (4, 8), generator=g))
    for _ in range(steps):
        if fixed_batch:
            idx, target = fixed
        else:
            idx = torch.randint(0, 17, (4, 8), generator=g)
            target = torch.randint(0, 17, (4, 8), generator=g)
        logits = model(idx)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 17), target.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def _check_manifold(model: Toy, optimizer: Dervish, cap: float | None, floor: float) -> None:
    names = {p: n for n, p in model.named_parameters()}
    for group in optimizer.param_groups:
        if group["role"] != "frame":
            continue
        for p in group["params"]:
            state = optimizer.state[p]
            frame, scale = state["frame"], state["scale"]
            assert float(orthonormality_error(frame).max()) < 1e-4, names[p]
            mat = matricize(p.detach(), optimizer.view_for(p))
            wide = mat.wide.reshape(-1, mat.m, mat.n).to(torch.float32)
            for i, w in enumerate(wide):
                sv = torch.linalg.svdvals(w)
                s_sorted = torch.sort(scale.reshape(-1, mat.m)[i], descending=True).values
                assert torch.allclose(sv, s_sorted, atol=1e-4), (names[p], i)
                if cap is not None:
                    assert float(sv.max()) <= cap + 1e-4, names[p]
                assert float(sv.min()) >= floor - 1e-5, names[p]


@pytest.mark.parametrize("cap,floor", [(None, 0.0), (1.0, 0.05)])
def test_dervish_keeps_the_factored_form_for_every_shape(cap, floor):
    torch.manual_seed(1)
    model = Toy()
    opt, report = build_dervish(
        model.named_parameters(),
        is_frame=_frame_predicate,
        views=_views(model),
        frame_kwargs={"lr": 0.1, "scale_lr": 1e-2, "scale_cap": cap, "scale_floor": floor},
    )
    assert report["frame_tensors"] == 6 and report["adam_tensors"] == 4
    opt.initialize()
    _check_manifold(model, opt, cap, floor)
    losses = _train(model, opt, steps=25, seed=2, fixed_batch=True)
    _check_manifold(model, opt, cap, floor)
    assert losses[-1] < losses[0]  # a fixed batch must be memorised
    rep = opt.frame_report({p: n for n, p in model.named_parameters()})
    assert len(rep) == 6
    assert max(r["orthonormality_error_max"] for r in rep) < 1e-4
    heads = next(r for r in rep if r["name"] == "blocks.attn.weight")
    assert heads["frames"] == 12 and heads["m"] == 8 and heads["n"] == 32


def test_spectral_cap_bounds_the_spectral_norm_under_pressure():
    # Push hard: large frame and scale learning rates, no decay. Muon's norms
    # should drift up; Dervish with a cap cannot exceed it, by construction.
    torch.manual_seed(3)
    cap = 0.5
    model_d = Toy()
    opt_d, _ = build_dervish(
        model_d.named_parameters(),
        is_frame=_frame_predicate,
        views=_views(model_d),
        frame_kwargs={"lr": 0.2, "scale_lr": 5e-2, "scale_cap": cap},
    )
    opt_d.initialize()
    _train(model_d, opt_d, steps=40, seed=4)
    report = hidden_matrix_report(model_d, _frame_predicate, _views(model_d))
    assert report["hidden_spectral_norm_max"] <= cap + 1e-4

    torch.manual_seed(3)
    model_m = Toy()
    opt_m, _ = build_muon(model_m.named_parameters(), is_muon=_frame_predicate, muon_kwargs={"lr": 0.2})
    before = spectral_norm(model_m.blocks["down"].weight.detach())
    _train(model_m, opt_m, steps=40, seed=4)
    after = spectral_norm(model_m.blocks["down"].weight.detach())
    assert after > before  # the control actually grows here


def test_scale_floor_prevents_rank_collapse():
    torch.manual_seed(5)
    model = Toy()
    opt, _ = build_dervish(
        model.named_parameters(),
        is_frame=_frame_predicate,
        views=_views(model),
        frame_kwargs={"lr": 0.1, "scale_lr": 1e-1, "scale_cap": 1.0, "scale_floor": 0.2},
    )
    opt.initialize()
    _train(model, opt, steps=30, seed=6)
    w = model.blocks["down"].weight.detach()
    # with s in [0.2, 1.0] the stable rank is at least m * (0.2/1.0)^2
    m = min(w.shape)
    assert stable_rank(w) >= m * 0.04 - 1e-6


def test_dervish_is_deterministic():
    def run():
        torch.manual_seed(8)
        model = Toy()
        opt, _ = build_dervish(model.named_parameters(), is_frame=_frame_predicate, views=_views(model))
        opt.initialize()
        _train(model, opt, steps=10, seed=9)
        return torch.cat([p.detach().flatten() for p in model.parameters()])

    assert torch.equal(run(), run())


def test_muon_reference_is_deterministic_and_descends():
    torch.manual_seed(10)
    model = Toy()
    opt, report = build_muon(model.named_parameters(), is_muon=_frame_predicate)
    assert report["muon_tensors"] == 6
    losses = _train(model, opt, steps=25, seed=11, fixed_batch=True)
    assert losses[-1] < losses[0]


def test_group_validation():
    p = nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError):
        Dervish([{"params": [p]}])
    with pytest.raises(ValueError):
        Dervish([{"params": [p], "role": "frame", "scale_cap": -1.0}])
    with pytest.raises(ValueError):
        Dervish([{"params": [nn.Parameter(torch.randn(4))], "role": "frame"}])
    with pytest.raises(ValueError):
        MuonWithAuxAdam([{"params": [p]}])


def test_rotation_interpretation_small_step_is_an_orthogonal_gate():
    # Square frame: one Dervish step maps Q to R Q with R orthogonal, angle ~ lr.
    # Wide frame: the rotation acts on the ambient space, so test the step size.
    torch.manual_seed(12)
    model = Toy()
    opt, _ = build_dervish(model.named_parameters(), is_frame=_frame_predicate, views=_views(model),
                           frame_kwargs={"lr": 0.05, "scale_lr": 0.0})
    opt.initialize()
    square = model.blocks["proj"].weight
    wide = model.blocks["down"].weight
    q0_square = opt.state[square]["frame"].clone()
    q0_wide = opt.state[wide]["frame"].clone()
    _train(model, opt, steps=1, seed=13)
    q1_square = opt.state[square]["frame"]
    q1_wide = opt.state[wide]["frame"]
    gate = q1_square @ q0_square.mT
    assert float(orthonormality_error(gate)) < 1e-4
    # Muon's quintic leaves singular values of the direction in about [0.68, 1.13],
    # so the per-step angle is at most about 1.2 * lr (docs/DESIGN.md states this).
    angle = float(torch.linalg.matrix_norm(gate - torch.eye(gate.size(0)), ord=2))
    assert 0.0 < angle <= 0.05 * 1.25
    step = float(torch.linalg.matrix_norm(q1_wide - q0_wide, ord=2))
    assert 0.0 < step <= 0.05 * 1.25


def test_frame_option_init_polar_iterative():
    torch.manual_seed(20)
    model = Toy()
    opt, _ = build_dervish(
        model.named_parameters(),
        is_frame=_frame_predicate,
        views=_views(model),
        frame_kwargs={"init_polar": "iterative"},
    )
    opt.initialize()
    _check_manifold(model, opt, cap=None, floor=0.0)
    for p in opt.frame_parameters():
        state = opt.state[p]
        mat = matricize(p.detach(), opt.view_for(p))
        expected_w = state["scale"].unsqueeze(-1) * state["frame"]
        assert torch.allclose(mat.wide, expected_w, atol=1e-4)


def test_frame_option_tangent_false():
    torch.manual_seed(21)
    cap = 0.8
    floor = 0.05
    model = Toy()
    opt, _ = build_dervish(
        model.named_parameters(),
        is_frame=_frame_predicate,
        views=_views(model),
        frame_kwargs={"tangent": False, "scale_cap": cap, "scale_floor": floor, "lr": 0.05, "scale_lr": 1e-2},
    )
    opt.initialize()
    _train(model, opt, steps=20, seed=22)
    _check_manifold(model, opt, cap=cap, floor=floor)
    report = hidden_matrix_report(model, _frame_predicate, _views(model))
    assert report["hidden_spectral_norm_max"] <= cap + 1e-4


def test_frame_option_scale_decay():
    def run_decay(decay: float) -> torch.Tensor:
        torch.manual_seed(23)
        model = Toy()
        opt, _ = build_dervish(
            model.named_parameters(),
            is_frame=_frame_predicate,
            views=_views(model),
            frame_kwargs={"scale_decay": decay, "scale_lr": 1e-2},
        )
        opt.initialize()
        _train(model, opt, steps=20, seed=24, fixed_batch=True)
        return torch.cat([opt.state[p]["scale"].flatten() for p in opt.frame_parameters()])

    s_no_decay = run_decay(0.0)
    s_decay = run_decay(0.1)
    assert torch.all(s_decay < s_no_decay)


def test_frame_option_retraction_steps():
    torch.manual_seed(25)
    model = Toy()
    opt, _ = build_dervish(
        model.named_parameters(),
        is_frame=_frame_predicate,
        views=_views(model),
        frame_kwargs={"lr": 0.2, "retraction_steps": 2},
    )
    opt.initialize()
    _train(model, opt, steps=30, seed=26)
    rep = opt.frame_report()
    assert max(r["orthonormality_error_max"] for r in rep) < 1e-4
    _check_manifold(model, opt, cap=None, floor=0.0)


def test_frame_option_ns_dtype_bfloat16():
    torch.manual_seed(27)
    cap = 0.75
    model = Toy()
    opt, _ = build_dervish(
        model.named_parameters(),
        is_frame=_frame_predicate,
        views=_views(model),
        frame_kwargs={"ns_dtype": torch.bfloat16, "scale_cap": cap, "lr": 0.05, "scale_lr": 1e-2},
    )
    opt.initialize()
    _train(model, opt, steps=20, seed=28)
    _check_manifold(model, opt, cap=cap, floor=0.0)
    report = hidden_matrix_report(model, _frame_predicate, _views(model))
    assert report["hidden_spectral_norm_max"] <= cap + 1e-4

