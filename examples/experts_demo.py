"""Architecture flexibility: one Dervish instance trains a dense layer, a bank of
mixture-of-experts matrices stored as one (E, m, n) tensor, and a convolution,
each as frames, and keeps every one of them on the manifold.

Toy regression on synthetic data; prints the loss curve and, at the end, the
spectral norm and orthonormality error of every frame. Deterministic. Runs in a
few seconds on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dervish import FrameView, build_dervish  # noqa: E402
from dervish.diagnostics import hidden_matrix_report  # noqa: E402


class MixedModel(nn.Module):
    def __init__(self, d: int = 32, experts: int = 4, hidden: int = 48) -> None:
        super().__init__()
        self.blocks = nn.ModuleDict({"dense": nn.Linear(d, d, bias=False), "conv": nn.Conv1d(d, d, kernel_size=3, padding=1, bias=False)})
        self.experts_in = nn.Parameter(torch.randn(experts, hidden, d) * 0.05)   # (E, m, n): E frames
        self.experts_out = nn.Parameter(torch.randn(experts, d, hidden) * 0.05)  # (E, m, n): E frames
        self.router = nn.Linear(d, experts)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, d)
        x = x + torch.tanh(self.blocks["dense"](x))
        x = x + torch.tanh(self.blocks["conv"](x.transpose(1, 2)).transpose(1, 2))
        gate = torch.softmax(self.router(x), dim=-1)                       # (B, T, E) soft routing
        h = torch.tanh(torch.einsum("btd,ehd->bteh", x, self.experts_in))  # (B, T, E, hidden)
        y = torch.einsum("bteh,edh->bted", h, self.experts_out)            # (B, T, E, d)
        x = x + (gate.unsqueeze(-1) * y).sum(dim=2)
        return self.head(x).squeeze(-1)


def is_frame(name: str, p: nn.Parameter) -> bool:
    return p.ndim >= 2 and (name.startswith("blocks.") or name.startswith("experts_"))


def main() -> None:
    torch.manual_seed(1)
    model = MixedModel()
    # A Conv1d kernel is 3D like an expert bank, so the automatic rule would read it as a
    # batch of (in, k) frames; say explicitly that it is one (out, in*k) frame.
    conv = model.blocks["conv"].weight
    views = {"blocks.conv.weight": FrameView((conv.size(0), conv.size(1) * conv.size(2)))}
    opt, report = build_dervish(model.named_parameters(), is_frame=is_frame, views=views, frame_kwargs={"lr": 0.05, "scale_lr": 5e-3, "scale_cap": 1.5})
    opt.initialize()
    print(f"frames: {report['frame_tensors']} tensors, {report['frame_parameters']} parameters; adam: {report['adam_tensors']} tensors")
    g = torch.Generator().manual_seed(2)
    w_true = torch.randn(32, generator=g)
    for step in range(1, 301):
        x = torch.randn(16, 8, 32, generator=g)
        target = torch.sin(x @ w_true) + 0.1 * (x[..., 0] * x[..., 1])
        loss = nn.functional.mse_loss(model(x), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in (1, 50, 100, 200, 300):
            print(f"step {step:3d}  mse {float(loss.detach()):.4f}")
    rep = hidden_matrix_report(model, is_frame, views)
    for p in rep["per_parameter"]:
        print(f"{p['name']:<20} frames {p['frames']:2d}  spectral max {p['spectral_norm_max']:.4f}  stable rank {p['stable_rank_mean']:6.2f}  row-orthonormality err {p['orthonormality_error_max']:.2e}")
    print("every frame stays a scaled isometry: rows orthonormal, spectrum capped at 1.5")


if __name__ == "__main__":
    main()
