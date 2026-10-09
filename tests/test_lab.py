"""Tests for the lab itself: the measurements are only worth what the harness is."""

from pathlib import Path

import torch

from lab.data import LAB_ALPHABET, build_codec, chunked_windows, normalize_natural_text, split_at_index
from lab.model import AttentionStats, TinyTransformer, is_hidden
from lab.train import answer_position_mask, weighted_loss

ROOT = Path(__file__).resolve().parents[1]


def test_model_is_causal():
    torch.manual_seed(0)
    model = TinyTransformer(vocab_size=36, block_size=64, n_layer=2, n_head=4, n_embd=32).eval()
    x = torch.randint(0, 36, (2, 64))
    x2 = x.clone()
    x2[:, 32:] = torch.randint(0, 36, (2, 32))
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x2)
    assert torch.equal(a[:, :32], b[:, :32])
    assert not torch.equal(a[:, 32:], b[:, 32:])


def test_chunked_windows_targets_are_next_characters():
    data = torch.arange(1000)
    x, y = chunked_windows(data, 32)
    assert x.shape == y.shape
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.equal(y[:-1, -1], x[1:, 0])


def test_split_keeps_validation_disjoint():
    n = 10_000
    split = split_at_index(n, 0.1, 128)
    assert 128 < split < n - 128
    assert split == int(n * 0.9)


def test_normalisation_is_deterministic_and_in_alphabet():
    raw = "First Citizen:\r\nBefore we proceed any further, hear me speak.\n\n\n\nAll:  Speak, speak."
    out = normalize_natural_text(raw)
    assert out == normalize_natural_text(raw)
    assert set(out) <= set(LAB_ALPHABET)
    assert "  " not in out and "\n\n\n" not in out
    codec = build_codec(out)
    assert codec.decode(codec.encode(out)) == out


def test_answer_mask_and_weighted_loss():
    x = torch.tensor([[5, 1, 2, 3, 9, 1, 2, 3, 7]])
    mask = answer_position_mask(x, [1, 2, 3])
    assert mask.int().tolist() == [[0, 0, 0, 1, 0, 0, 0, 1, 0]]
    logits = torch.randn(1, 9, 11)
    y = torch.randint(0, 11, (1, 9))
    plain = torch.nn.functional.cross_entropy(logits.view(-1, 11), y.view(-1))
    assert torch.allclose(weighted_loss(logits, y, mask, 1.0), plain)
    heavy = weighted_loss(logits, y, mask, 50.0)
    per = torch.nn.functional.cross_entropy(logits.view(-1, 11), y.view(-1), reduction="none").view(1, 9)
    expected = (per * (1 + 49 * mask.float())).sum() / (1 + 49 * mask.float()).sum()
    assert torch.allclose(heavy, expected)


def test_attention_stats_and_bound_are_consistent():
    torch.manual_seed(1)
    model = TinyTransformer(vocab_size=36, block_size=32, n_layer=2, n_head=4, n_embd=32).eval()
    stats = AttentionStats()
    model.set_stats(stats)
    with torch.no_grad():
        model(torch.randint(0, 36, (3, 32)))
    model.set_stats(None)
    assert len(stats.logit_max) == 2 and len(stats.entropy_mean) == 2
    # The bound with the actual per-head spectral norms must dominate the measurement.
    caps = []
    for name, p in model.named_parameters():
        if name.endswith("c_attn.weight"):
            w = p.detach().view(3 * 4, 8, 32)
            caps.append(float(torch.linalg.matrix_norm(w, ord=2).max()))
    cap = max(caps)
    assert max(stats.logit_max) <= model.attention_logit_bound(cap, cap)


def test_hidden_predicate_matches_reference_selection():
    model = TinyTransformer(vocab_size=36, block_size=16, n_layer=1, n_head=2, n_embd=16)
    hidden = [n for n, p in model.named_parameters() if is_hidden(n, p)]
    assert hidden == ["blocks.0.attn.c_attn.weight", "blocks.0.attn.c_proj.weight", "blocks.0.mlp.0.weight", "blocks.0.mlp.2.weight"]
