# Plugging Dervish into an existing trainer

A typical Muon trainer selects its optimizer in a function such as
`build_optimizer(model, args)`. Dervish slots in as a third branch beside
`adamw` and `muon`. Nothing in the model changes, because Dervish writes plain
weight tensors back into the parameters.

## 1 · Make the package importable

Copy `dervish/` next to the trainer, or add this repository's root to
`PYTHONPATH`.

## 2 · The branch

In `build_optimizer`, after the `muon` branch (the parameter names below assume
a GPT-style model with a fused `c_attn` projection per block):

```python
from dervish import FrameView, build_dervish

if args.optimizer == "dervish":
    head_size = args.n_embd // args.n_head
    views = {
        f"blocks.{i}.attn.c_attn.weight": FrameView.heads(3 * args.n_head, head_size, args.n_embd)
        for i in range(args.n_layer)
    }
    optimizer, report = build_dervish(
        trainable_named_parameters,
        is_frame=lambda name, p: p.ndim >= 2 and name.startswith("blocks."),
        views=views,
        frame_kwargs={
            "lr": args.dervish_lr,                 # rotation angle per step; 0.05 in the lab
            "momentum": args.muon_momentum,
            "ns_steps": args.muon_ns_steps,
            "nesterov": not args.no_muon_nesterov,
            "ns_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
            "scale_lr": args.dervish_scale_lr,     # 1e-3 in the lab
            "scale_cap": args.dervish_cap,         # 1.0 in the lab; None for unbounded
            "scale_floor": args.dervish_floor,     # 0.0 in the lab
        },
        adam_kwargs={
            "lr": args.lr,
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_eps,
            "weight_decay": args.weight_decay,
        },
    )
    optimizer.initialize()   # project the hidden matrices onto the manifold before step 0
    return optimizer, {"optimizer": "dervish", **report}
```

and the four flags in the parser:

```python
parser.add_argument("--dervish-lr", type=float, default=0.05)
parser.add_argument("--dervish-scale-lr", type=float, default=1e-3)
parser.add_argument("--dervish-cap", type=float, default=1.0)
parser.add_argument("--dervish-floor", type=float, default=0.0)
```

plus `"dervish"` in the `--optimizer` choices. If the trainer wraps its linears
(LoRA, adapters), point the predicate at the base weights or route the adapter
matrices to Adam: a rank-2 `lora_b` of shape `(n, 2)` is a legal tall frame of
two orthonormal columns, but probably not what an adapter wants.

## 3 · Checkpoints

Dervish's state (`frame`, `scale`, `momentum`, the spectrum's Adam moments)
lives in `optimizer.state_dict()` like Muon's momentum buffer and resumes the
same way. The `FrameView` map is not serialised; pass it again at
construction.

## 4 · The stage to run first

The hypothesis's "what would change the plan" section names it: E1 on a GPU
trainer at about `25M` parameters and `5,000` steps, seeds `7 11 19`, `muon`
versus `dervish`, on CUDA with the bfloat16 Newton-Schulz path. Watch
`att_logit_max`; if the trainer does not log it, the hook in
`lab/model.py: CausalSelfAttention.forward` is ten lines and carries over.
