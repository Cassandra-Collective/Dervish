import pytest
import torch

from dervish.matricize import FrameView, matricize, unmatricize, viewed_shape_for


def test_auto_rules():
    assert viewed_shape_for((64, 128), FrameView.auto()) == (64, 128)
    assert viewed_shape_for((4, 64, 128), FrameView.auto()) == (4, 64, 128)
    assert viewed_shape_for((16, 8, 3, 3), FrameView.auto()) == (16, 72)
    with pytest.raises(ValueError):
        viewed_shape_for((128,), FrameView.auto())


def test_head_view_splits_fused_projection():
    n_embd, n_head = 64, 4
    w = torch.randn(3 * n_embd, n_embd)
    view = FrameView.heads(3 * n_head, n_embd // n_head, n_embd)
    mat = matricize(w, view)
    assert mat.wide.shape == (12, 16, 64)
    assert not mat.transposed
    back = unmatricize(mat.wide, mat, tuple(w.shape))
    assert torch.equal(back, w)


def test_tall_matrix_is_transposed_and_restored():
    w = torch.randn(256, 64)
    mat = matricize(w, FrameView.auto())
    assert mat.transposed and mat.wide.shape == (64, 256)
    assert torch.equal(unmatricize(mat.wide, mat, (256, 64)), w)


def test_wrong_view_fails_loudly():
    with pytest.raises(ValueError):
        matricize(torch.randn(10, 10), FrameView((3, 5, 7)))
