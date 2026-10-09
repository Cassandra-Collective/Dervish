"""Shape rules: how a parameter tensor becomes one or more frames.

Dervish and Muon act on matrices. Real models carry other shapes, and the
mapping from "parameter" to "matrix" is a modelling decision, not a detail:

- a 2D weight is one frame;
- a 3D tensor (E, m, n) is E independent frames, which is how mixture-of-experts
  expert banks are stored;
- a 4D convolution kernel (out, in, kh, kw) is one frame (out, in*kh*kw);
- a fused projection such as the attention c_attn weight (3*n_embd, n_embd) can
  be viewed as (3*n_head, head_dim, n_embd), one frame per head and per role,
  so that every head's query, key and value projection is its own scaled
  isometry. That is the view the lab uses for attention.

`FrameView` carries the rule; `matricize` applies it and returns a wide
(..., m, n) view plus the bookkeeping needed to write the result back with
`unmatricize`. Views never copy; writes go through `copy_` into the parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrameView:
    """How to see one parameter as frames.

    `shape` is the batched-matrix shape (..., m, n) the parameter is viewed as.
    `None` means the automatic rule: 2D as is, 3D as a batch, 4D conv as
    (out, in*kh*kw).
    """

    shape: tuple[int, ...] | None = None

    @staticmethod
    def auto() -> "FrameView":
        return FrameView(None)

    @staticmethod
    def heads(n_blocks: int, block_rows: int, n_in: int) -> "FrameView":
        """A fused (n_blocks*block_rows, n_in) projection as n_blocks frames."""
        return FrameView((n_blocks, block_rows, n_in))


@dataclass(frozen=True)
class Matricized:
    """A wide (..., m, n) view of a parameter plus how to undo it."""

    wide: torch.Tensor
    viewed_shape: tuple[int, ...]
    transposed: bool

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.wide.shape[:-2])

    @property
    def m(self) -> int:
        return int(self.wide.size(-2))

    @property
    def n(self) -> int:
        return int(self.wide.size(-1))


def viewed_shape_for(param_shape: tuple[int, ...], view: FrameView) -> tuple[int, ...]:
    if view.shape is not None:
        if _numel(view.shape) != _numel(param_shape):
            raise ValueError(f"FrameView shape {view.shape} does not match parameter shape {param_shape}")
        if len(view.shape) < 2:
            raise ValueError("FrameView shape needs at least two dimensions")
        return tuple(view.shape)
    if len(param_shape) == 2:
        return tuple(param_shape)
    if len(param_shape) == 3:
        return tuple(param_shape)
    if len(param_shape) == 4:
        out_channels = param_shape[0]
        return (out_channels, _numel(param_shape[1:]))
    raise ValueError(
        f"No automatic frame rule for a parameter of shape {param_shape}; pass an explicit FrameView"
    )


def matricize(tensor: torch.Tensor, view: FrameView) -> Matricized:
    """View `tensor` (a parameter or its gradient) as wide frames."""
    shape = viewed_shape_for(tuple(tensor.shape), view)
    viewed = tensor.reshape(shape)
    transposed = viewed.size(-2) > viewed.size(-1)
    wide = viewed.mT if transposed else viewed
    return Matricized(wide=wide, viewed_shape=shape, transposed=transposed)


def unmatricize(wide: torch.Tensor, like: Matricized, param_shape: tuple[int, ...]) -> torch.Tensor:
    """Inverse of `matricize`: a tensor shaped like the original parameter."""
    viewed = wide.mT if like.transposed else wide
    return viewed.reshape(param_shape)


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total
