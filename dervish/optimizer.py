"""Dervish, and a reference Muon implementation.

Dervish keeps every frame parameter W in the form W = diag(s) Q with Q
orthonormal (rows if W is wide, columns if tall; internally everything is
handled wide). One step:

  1. split the gradient G of W into a spectrum gradient g_s = rowsum(G * Q)
     and a frame gradient G_Q = diag(s) G;
  2. Muon-style momentum on G_Q, projected onto the tangent space of the
     Stiefel manifold at Q, then orthogonalised with the quintic
     Newton-Schulz iteration (unit spectral norm, every direction alive);
  3. retraction: Q <- polar(Q - lr * U) with two cubic Newton-Schulz steps.
     Because U is tangent, Q - lr U has singular values in
     [1, sqrt(1 + lr^2)], so the polar step is cheap and exact to fp32. The
     result is Q rotated by a small orthogonal matrix: one "gate" per step;
  4. Adam on s with its own learning rate, optional decoupled decay, and a
     hard clamp to [scale_floor, scale_cap];
  5. write W = diag(s) Q back into the parameter.

Everything that is not a frame (embeddings, output head, norms, biases) is
trained by the same auxiliary Adam that Muon uses, so a Dervish-vs-Muon
comparison changes exactly one thing: how hidden matrices move.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn

from .matricize import FrameView, Matricized, matricize, unmatricize
from .polar import orthonormality_error, polar, polar_cubic, polar_exact, tangent_project, zeropower_quintic

# ----------------------------------------------------------------------------
# Shared Adam step (mirrors a reference Muon trainer's adam_update exactly)
# ----------------------------------------------------------------------------


def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    exp_avg.lerp_(grad, 1.0 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1.0 - betas[1])
    exp_avg_corrected = exp_avg / (1.0 - betas[0] ** step)
    exp_avg_sq_corrected = exp_avg_sq / (1.0 - betas[1] ** step)
    return exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + eps)


# ----------------------------------------------------------------------------
# Muon reference (mirrors the common SingleDeviceMuonWithAuxAdam)
# ----------------------------------------------------------------------------


def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum: float,
    ns_steps: int,
    nesterov: bool,
    ns_dtype: torch.dtype,
) -> torch.Tensor:
    momentum_buffer.lerp_(grad, 1.0 - momentum)
    update = grad.lerp(momentum_buffer, momentum) if nesterov else momentum_buffer
    original_shape = update.shape
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_quintic(update, ns_steps, dtype=ns_dtype)
    update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    return update.reshape(original_shape).to(dtype=grad.dtype)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon with an auxiliary Adam, the common single-device recipe, line for line.

    Param groups carry `use_muon`. Muon groups: lr 0.02, momentum 0.95,
    weight_decay 0.0, ns_steps 5, nesterov True, ns_dtype float32. Adam groups:
    lr 3e-4, betas (0.9, 0.95), eps 1e-10, weight_decay 0.0. The only addition
    over the reference code is `ns_dtype` (the original hard-codes bfloat16).
    """

    def __init__(self, param_groups: list[dict[str, Any]]) -> None:
        if not param_groups:
            raise ValueError("Muon optimizer received no parameter groups")
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Muon parameter groups require use_muon")
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda parameter: parameter.numel(), reverse=True)
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("ns_steps", 5)
                group.setdefault("nesterov", True)
                group.setdefault("ns_dtype", torch.float32)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    update = muon_update(
                        parameter.grad,
                        state["momentum_buffer"],
                        float(group["momentum"]),
                        int(group["ns_steps"]),
                        bool(group["nesterov"]),
                        group["ns_dtype"],
                    )
                    parameter.mul_(1.0 - float(group["lr"]) * float(group["weight_decay"]))
                    parameter.add_(update, alpha=-float(group["lr"]))
            else:
                _adam_group_step(self.state, group)
        return loss


def _adam_group_step(state_store: Any, group: dict[str, Any]) -> None:
    for parameter in group["params"]:
        if parameter.grad is None:
            continue
        state = state_store[parameter]
        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] = 0
        state["step"] += 1
        update = adam_update(
            parameter.grad,
            state["exp_avg"],
            state["exp_avg_sq"],
            int(state["step"]),
            group["betas"],
            float(group["eps"]),
        )
        parameter.mul_(1.0 - float(group["lr"]) * float(group["weight_decay"]))
        parameter.add_(update, alpha=-float(group["lr"]))


# ----------------------------------------------------------------------------
# Dervish
# ----------------------------------------------------------------------------

FRAME_DEFAULTS: dict[str, Any] = {
    "lr": 0.05,  # rotation angle per step, in radians, since U has unit spectral norm
    "momentum": 0.95,
    "nesterov": True,
    "ns_steps": 5,
    "ns_dtype": torch.float32,
    "retraction_steps": 2,
    "tangent": True,
    "scale_lr": 1e-3,
    "scale_betas": (0.9, 0.95),
    "scale_eps": 1e-8,
    "scale_decay": 0.0,
    "scale_cap": None,  # upper bound on every singular value; None = unbounded
    "scale_floor": 0.0,  # lower bound on every singular value
    "init_polar": "svd",  # "svd" (exact, once per parameter) or "iterative"
}

ADAM_DEFAULTS: dict[str, Any] = {
    "lr": 3e-4,
    "betas": (0.9, 0.95),
    "eps": 1e-10,
    "weight_decay": 0.0,
}


class Dervish(torch.optim.Optimizer):
    """Rotation-based successor to Muon. See the module docstring.

    `param_groups` entries carry `role`: "frame" or "adam". Frame groups accept
    the keys in FRAME_DEFAULTS; adam groups the keys in ADAM_DEFAULTS. `views`
    maps a frame parameter to the FrameView that says how it is cut into
    matrices (default: FrameView.auto()).
    """

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        views: dict[nn.Parameter, FrameView] | None = None,
    ) -> None:
        if not param_groups:
            raise ValueError("Dervish received no parameter groups")
        for group in param_groups:
            role = group.get("role")
            if role not in {"frame", "adam"}:
                raise ValueError("Dervish parameter groups require role='frame' or role='adam'")
            defaults = FRAME_DEFAULTS if role == "frame" else ADAM_DEFAULTS
            for key, value in defaults.items():
                group.setdefault(key, value)
            if role == "frame":
                cap = group["scale_cap"]
                floor = group["scale_floor"]
                if cap is not None and cap <= 0:
                    raise ValueError("scale_cap must be positive")
                if floor < 0:
                    raise ValueError("scale_floor must be non-negative")
                if cap is not None and floor > cap:
                    raise ValueError("scale_floor must not exceed scale_cap")
        super().__init__(param_groups, {})
        self._views: dict[int, FrameView] = {}
        for parameter, view in (views or {}).items():
            self._views[id(parameter)] = view
        for group in self.param_groups:
            if group["role"] == "frame":
                for parameter in group["params"]:
                    if parameter.ndim < 2:
                        raise ValueError("frame parameters need at least two dimensions")
                    # validate the view eagerly so shape mistakes fail at construction
                    matricize(parameter.detach(), self.view_for(parameter))

    # -- views and state ------------------------------------------------------

    def view_for(self, parameter: nn.Parameter) -> FrameView:
        return self._views.get(id(parameter), FrameView.auto())

    def frame_parameters(self) -> list[nn.Parameter]:
        return [p for g in self.param_groups if g["role"] == "frame" for p in g["params"]]

    @torch.no_grad()
    def initialize(self) -> None:
        """Project every frame parameter onto the manifold now (W <- diag(s) polar(W)).

        Called lazily by `step` otherwise. Calling it before the first forward
        pass makes the step-0 evaluation see the same model training sees.
        """
        for group in self.param_groups:
            if group["role"] != "frame":
                continue
            for parameter in group["params"]:
                if len(self.state[parameter]) == 0:
                    self._init_frame(parameter, group)

    def _init_frame(self, parameter: nn.Parameter, group: dict[str, Any]) -> None:
        mat = matricize(parameter.detach(), self.view_for(parameter))
        wide = mat.wide.to(torch.float32)
        scale = wide.norm(dim=-1)
        frame = polar_exact(wide) if group["init_polar"] == "svd" else polar(wide)
        scale = _clamp_scale(scale, group)
        state = self.state[parameter]
        state["frame"] = frame.contiguous()
        state["scale"] = scale.contiguous()
        state["momentum"] = torch.zeros_like(frame)
        state["scale_exp_avg"] = torch.zeros_like(scale)
        state["scale_exp_avg_sq"] = torch.zeros_like(scale)
        state["step"] = 0
        self._write_back(parameter, mat, state)

    @staticmethod
    def _write_back(parameter: nn.Parameter, mat: Matricized, state: dict[str, Any]) -> None:
        wide = state["scale"].unsqueeze(-1) * state["frame"]
        parameter.copy_(unmatricize(wide, mat, tuple(parameter.shape)).to(parameter.dtype))

    # -- the step -------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["role"] == "frame":
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    if len(self.state[parameter]) == 0:
                        self._init_frame(parameter, group)
                    self._frame_step(parameter, group)
            else:
                _adam_group_step(self.state, group)
        return loss

    def _frame_step(self, parameter: nn.Parameter, group: dict[str, Any]) -> None:
        state = self.state[parameter]
        view = self.view_for(parameter)
        mat = matricize(parameter.detach(), view)
        grad = matricize(parameter.grad, view).wide.to(torch.float32)
        frame: torch.Tensor = state["frame"]
        scale: torch.Tensor = state["scale"]

        # 1. split the gradient
        scale_grad = (grad * frame).sum(dim=-1)
        frame_grad = scale.unsqueeze(-1) * grad

        # 2. momentum, tangent projection, orthogonalisation
        momentum = float(group["momentum"])
        buffer: torch.Tensor = state["momentum"]
        buffer.lerp_(frame_grad, 1.0 - momentum)
        direction = frame_grad.lerp(buffer, momentum) if group["nesterov"] else buffer.clone()
        if group["tangent"]:
            direction = tangent_project(direction, frame)
        update = zeropower_quintic(direction, int(group["ns_steps"]), dtype=group["ns_dtype"]).to(torch.float32)
        if group["tangent"]:
            update = tangent_project(update, frame)

        # 3. rotate the frame: retraction by the cubic polar iteration
        new_frame = polar_cubic(frame - float(group["lr"]) * update, steps=int(group["retraction_steps"]))
        frame.copy_(new_frame)

        # 4. Adam on the spectrum, then clamp
        state["step"] += 1
        scale_update = adam_update(
            scale_grad,
            state["scale_exp_avg"],
            state["scale_exp_avg_sq"],
            int(state["step"]),
            tuple(group["scale_betas"]),
            float(group["scale_eps"]),
        )
        scale_lr = float(group["scale_lr"])
        scale.mul_(1.0 - scale_lr * float(group["scale_decay"]))
        scale.add_(scale_update, alpha=-scale_lr)
        scale.copy_(_clamp_scale(scale, group))

        # 5. write W = diag(s) Q back
        self._write_back(parameter, mat, state)

    # -- diagnostics ----------------------------------------------------------

    @torch.no_grad()
    def frame_report(self, names: dict[nn.Parameter, str] | None = None) -> list[dict[str, Any]]:
        """Per frame parameter: shape, spectrum summary and orthonormality error."""
        report: list[dict[str, Any]] = []
        for group in self.param_groups:
            if group["role"] != "frame":
                continue
            for parameter in group["params"]:
                state = self.state[parameter]
                if len(state) == 0:
                    continue
                frame: torch.Tensor = state["frame"]
                scale: torch.Tensor = state["scale"]
                report.append(
                    {
                        "name": (names or {}).get(parameter, ""),
                        "param_shape": tuple(parameter.shape),
                        "frames": int(frame.numel() // (frame.size(-2) * frame.size(-1))),
                        "m": int(frame.size(-2)),
                        "n": int(frame.size(-1)),
                        "scale_min": float(scale.min()),
                        "scale_max": float(scale.max()),
                        "scale_mean": float(scale.mean()),
                        "orthonormality_error_max": float(orthonormality_error(frame).max()),
                    }
                )
        return report


def _clamp_scale(scale: torch.Tensor, group: dict[str, Any]) -> torch.Tensor:
    floor = float(group["scale_floor"])
    cap = group["scale_cap"]
    out = scale.clamp(min=floor)
    if cap is not None:
        out = out.clamp(max=float(cap))
    return out


# ----------------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------------


def default_is_frame(name: str, parameter: nn.Parameter) -> bool:
    """The usual Muon selection: hidden matrices (2D or more, inside blocks)."""
    return parameter.ndim >= 2 and name.startswith("blocks.")


def build_dervish(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    is_frame: Callable[[str, nn.Parameter], bool] = default_is_frame,
    views: dict[str, FrameView] | None = None,
    frame_kwargs: dict[str, Any] | None = None,
    adam_kwargs: dict[str, Any] | None = None,
) -> tuple[Dervish, dict[str, Any]]:
    """Split parameters into a frame group and an Adam group and build Dervish.

    Returns the optimizer and a small report (counts) in the style of the
    reference `build_optimizer` report dict.
    """
    frame_params: list[nn.Parameter] = []
    adam_params: list[nn.Parameter] = []
    frame_names: list[str] = []
    adam_names: list[str] = []
    param_views: dict[nn.Parameter, FrameView] = {}
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if is_frame(name, parameter):
            frame_params.append(parameter)
            frame_names.append(name)
            if views and name in views:
                param_views[parameter] = views[name]
        else:
            adam_params.append(parameter)
            adam_names.append(name)
    groups: list[dict[str, Any]] = []
    if frame_params:
        groups.append({"params": frame_params, "role": "frame", **(frame_kwargs or {})})
    if adam_params:
        groups.append({"params": adam_params, "role": "adam", **(adam_kwargs or {})})
    optimizer = Dervish(groups, views=param_views)
    return optimizer, {
        "optimizer": "dervish",
        "frame_parameters": sum(p.numel() for p in frame_params),
        "adam_parameters": sum(p.numel() for p in adam_params),
        "frame_tensors": len(frame_params),
        "adam_tensors": len(adam_params),
        "frame_names": frame_names,
        "adam_names": adam_names,
    }


def build_muon(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    is_muon: Callable[[str, nn.Parameter], bool] = default_is_frame,
    muon_kwargs: dict[str, Any] | None = None,
    adam_kwargs: dict[str, Any] | None = None,
) -> tuple[MuonWithAuxAdam, dict[str, Any]]:
    """The usual Muon split (hidden matrices to Muon, the rest to Adam)."""
    muon_params: list[nn.Parameter] = []
    adam_params: list[nn.Parameter] = []
    muon_names: list[str] = []
    adam_names: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if is_muon(name, parameter):
            muon_params.append(parameter)
            muon_names.append(name)
        else:
            adam_params.append(parameter)
            adam_names.append(name)
    groups: list[dict[str, Any]] = []
    if muon_params:
        groups.append({"params": muon_params, "use_muon": True, **(muon_kwargs or {})})
    if adam_params:
        groups.append({"params": adam_params, "use_muon": False, **(adam_kwargs or {})})
    optimizer = MuonWithAuxAdam(groups)
    return optimizer, {
        "optimizer": "muon",
        "muon_parameters": sum(p.numel() for p in muon_params),
        "adam_parameters": sum(p.numel() for p in adam_params),
        "muon_tensors": len(muon_params),
        "adam_tensors": len(adam_params),
        "muon_names": muon_names,
        "adam_names": adam_names,
    }
