"""Corpora, codec, splits and batches. Deterministic by construction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch

# The lab alphabet: 26 letters, space, newline and eight punctuation marks (36 symbols).
# The alphabet keeps
# the newline because Shakespeare's line structure carries information.
LAB_ALPHABET = "abcdefghijklmnopqrstuvwxyz \n.,;:!?'-"


@dataclass(frozen=True)
class Codec:
    chars: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[idx] for idx in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)


def build_codec(text: str) -> Codec:
    chars = sorted(set(text))
    return Codec(chars=chars, stoi={ch: i for i, ch in enumerate(chars)}, itos={i: ch for i, ch in enumerate(chars)})


def normalize_natural_text(text: str) -> str:
    """Lowercase, keep the lab alphabet, collapse runs of spaces. Deterministic."""
    text = text.lower().replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("’", "'").replace("‘", "'").replace("“", "'").replace("”", "'")
    allowed = set(LAB_ALPHABET)
    kept = "".join(ch for ch in text if ch in allowed)
    kept = re.sub(r" {2,}", " ", kept)
    kept = re.sub(r"\n{3,}", "\n\n", kept)
    return kept


def load_corpus(path: Path, normalize: bool) -> str:
    text = path.read_text(encoding="utf-8")
    if normalize:
        text = normalize_natural_text(text)
    if len(text) < 1000:
        raise ValueError(f"Corpus is too short: {path}")
    return text


def split_at_index(length: int, val_fraction: float, block_size: int) -> int:
    if not 0.05 <= val_fraction <= 0.5:
        raise ValueError("val_fraction must be between 0.05 and 0.5")
    split_at = max(block_size + 2, int(length * (1.0 - val_fraction)))
    return min(split_at, length - block_size - 2)


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, generator: torch.Generator, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    num_starts = len(data) - block_size
    if num_starts <= 0:
        raise ValueError("Data split is too short for the requested block size")
    starts = torch.randint(num_starts, (batch_size,), generator=generator, device=device)
    x = torch.stack([data[int(i) : int(i) + block_size] for i in starts])
    y = torch.stack([data[int(i) + 1 : int(i) + block_size + 1] for i in starts])
    return x, y


def chunked_windows(data: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-overlapping windows covering the split: the deterministic 'full' eval."""
    usable = (len(data) - 1) // block_size * block_size
    x = data[:usable].view(-1, block_size)
    y = data[1 : usable + 1].view(-1, block_size)
    return x, y
