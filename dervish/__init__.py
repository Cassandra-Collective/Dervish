"""Dervish · a rotation-based successor to Muon.

Every hidden weight matrix is kept, at all times, in the factored form

    W = diag(s) · Q        (Q has orthonormal rows; or the transpose for tall W)

The frame Q moves only by rotations (one small orthogonal "gate" per step,
found with Newton-Schulz polar iteration) and the spectrum s is a separate,
small, bounded vector trained with Adam. The singular values of W are |s|
exactly, so spectral norms, attention logits and output logits carry
provable bounds, and no singular direction can silently collapse.

See docs/DESIGN.md for the algorithm and the bounds, docs/PRIOR_ART.md for
what is and is not new here, and docs/RESULTS.md for the measurements.
"""

from .optimizer import Dervish, MuonWithAuxAdam, build_dervish, build_muon
from .polar import polar, polar_cubic, zeropower_quintic, orthonormality_error
from .matricize import FrameView, matricize, unmatricize
from . import diagnostics

__all__ = [
    "Dervish",
    "MuonWithAuxAdam",
    "build_dervish",
    "build_muon",
    "polar",
    "polar_cubic",
    "zeropower_quintic",
    "orthonormality_error",
    "FrameView",
    "matricize",
    "unmatricize",
    "diagnostics",
]

__version__ = "0.1.0"
