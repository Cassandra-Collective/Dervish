"""A self-contained, CPU-sized character-level language-model lab.

It exists so the optimizer runs anywhere with PyTorch and nothing else, and so a
Dervish-vs-Muon-vs-AdamW comparison changes exactly one thing. The model is
a small GPT-style `TinyTransformer` (pre-norm LayerNorm,
GELU MLP with biases, learned or RoPE positions, N(0, 0.02) init, untied
head).
"""
