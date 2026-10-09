"""TinyTransformer mirror with instrumentation for the claims under test."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from dervish.matricize import FrameView


@dataclass
class AttentionStats:
    """Collected during instrumented forward passes (evaluation only)."""

    logit_max: list[float] = field(default_factory=list)
    entropy_mean: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.logit_max.clear()
        self.entropy_mean.clear()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos = cos[:, :, : x.size(-2), :]
    sin = sin[:, :, : x.size(-2), :]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, pos_encoding: str) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        head_size = n_embd // n_head
        if pos_encoding == "rope" and head_size % 2 != 0:
            raise ValueError("rope requires an even head size")
        if pos_encoding not in {"learned", "rope"}:
            raise ValueError(f"Unknown positional encoding: {pos_encoding}")
        self.n_head = n_head
        self.pos_encoding = pos_encoding
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))
        if pos_encoding == "rope":
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_size, 2).float() / head_size))
            positions = torch.arange(block_size, dtype=torch.float32)
            freqs = torch.outer(positions, inv_freq)
            self.register_buffer("rope_cos", freqs.cos().view(1, 1, block_size, head_size // 2))
            self.register_buffer("rope_sin", freqs.sin().view(1, 1, block_size, head_size // 2))
        else:
            self.register_buffer("rope_cos", torch.empty(0), persistent=False)
            self.register_buffer("rope_sin", torch.empty(0), persistent=False)
        self.stats: AttentionStats | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        q, k, v = self.c_attn(x).split(channels, dim=2)
        head_size = channels // self.n_head
        q = q.view(batch, tokens, self.n_head, head_size).transpose(1, 2)
        k = k.view(batch, tokens, self.n_head, head_size).transpose(1, 2)
        v = v.view(batch, tokens, self.n_head, head_size).transpose(1, 2)
        if self.pos_encoding == "rope":
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        causal = self.mask[:, :, :tokens, :tokens] == 0
        if self.stats is not None:
            visible = att.masked_fill(causal, float("-inf"))
            self.stats.logit_max.append(float(att.masked_fill(causal, float("-inf")).max()))
            probs = F.softmax(visible, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
            self.stats.entropy_mean.append(float(entropy.mean()))
        att = att.masked_fill(causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, tokens, channels)
        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, pos_encoding: str) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, pos_encoding)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=True),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float = 0.0,
        pos_encoding: str = "rope",
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.n_head = n_head
        self.n_embd = n_embd
        self.pos_encoding = pos_encoding
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd) if pos_encoding == "learned" else None
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout, pos_encoding) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, tokens = idx.shape
        if tokens > self.block_size:
            raise ValueError("sequence longer than block_size")
        x = self.token_embedding(idx)
        if self.position_embedding is not None:
            positions = torch.arange(tokens, device=idx.device)
            x = x + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # -- instrumentation ------------------------------------------------------

    def set_stats(self, stats: AttentionStats | None) -> None:
        for block in self.blocks:
            block.attn.stats = stats

    def head_views(self) -> dict[str, FrameView]:
        """Per-head frames for every fused qkv projection (3 roles x n_head)."""
        head_size = self.n_embd // self.n_head
        return {
            f"blocks.{i}.attn.c_attn.weight": FrameView.heads(3 * self.n_head, head_size, self.n_embd)
            for i in range(len(self.blocks))
        }

    @torch.no_grad()
    def attention_logit_bound(self, q_cap: float, k_cap: float) -> float:
        """Analytic bound on any attention logit when per-head q and k frames are capped.

        With LayerNorm before attention, ||ln(x)|| <= sqrt(n_embd) * max|gain| + ||beta||
        per token. q = W_q ln(x) + b_q with ||W_q||_2 <= q_cap, and the same for k,
        so q.k / sqrt(d_head) <= (q_cap * r + ||b_q||)(k_cap * r + ||b_k||) / sqrt(d_head)
        with r the LayerNorm output norm bound. RoPE is a rotation and changes no norm.
        The bound is reported, not enforced, and holds per block; the max is returned.
        """
        head_size = self.n_embd // self.n_head
        worst = 0.0
        for block in self.blocks:
            gain = float(block.ln_1.weight.abs().max())
            beta = float(block.ln_1.bias.norm())
            r = math.sqrt(self.n_embd) * gain + beta
            bias = block.attn.c_attn.bias.view(3, self.n_head, head_size)
            bq = float(bias[0].norm(dim=-1).max())
            bk = float(bias[1].norm(dim=-1).max())
            worst = max(worst, (q_cap * r + bq) * (k_cap * r + bk) / math.sqrt(head_size))
        return worst


def is_hidden(name: str, parameter: nn.Parameter) -> bool:
    """The usual Muon selection: matrices inside the blocks."""
    return parameter.ndim >= 2 and name.startswith("blocks.")


def is_hidden_or_head(name: str, parameter: nn.Parameter) -> bool:
    return is_hidden(name, parameter) or name == "lm_head.weight"
