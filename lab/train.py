"""One training run: one corpus, one optimizer, one seed, one report dict.

The report is flat where a comparison runner expects flat fields (val NLL,
bits/char, parameter counts, seconds) and carries a `history` list with one
record per evaluation, which is where the logit, spectrum and copy-probe
curves live.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dervish import FrameView, build_dervish, build_muon  # noqa: E402
from dervish.diagnostics import hidden_matrix_report  # noqa: E402
from lab.data import build_codec, chunked_windows, get_batch, load_corpus, split_at_index  # noqa: E402
from lab.model import AttentionStats, TinyTransformer, is_hidden, is_hidden_or_head  # noqa: E402

LN2 = math.log(2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dervish lab: train one tiny character transformer")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--normalize", action="store_true", help="lowercase and restrict to the lab alphabet (natural text)")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave as is)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--pos-encoding", default="rope", choices=["rope", "learned"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--optimizer", default="dervish", choices=["adamw", "muon", "dervish"])
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (whole model for adamw; embeddings, head, norms, biases otherwise)")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="decoupled weight decay for adamw and for the Muon group")
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--no-nesterov", action="store_true")
    parser.add_argument("--mona-beta", type=float, default=-1.0, help="MONA acceleration EMA rate beta_a for the Muon or Dervish frame group (arXiv:2605.26842); < 0 means off")
    parser.add_argument("--mona-alpha", type=float, default=None, help="MONA acceleration coefficient; default: the paper's tie -1 / (2 (1 - beta_a))")
    parser.add_argument("--dervish-lr", type=float, default=0.05, help="rotation angle per step (radians)")
    parser.add_argument("--dervish-scale-lr", type=float, default=1e-3)
    parser.add_argument("--dervish-cap", type=float, default=1.0, help="upper bound on every singular value; <= 0 means unbounded")
    parser.add_argument("--dervish-floor", type=float, default=0.0)
    parser.add_argument("--dervish-scale-decay", type=float, default=0.0)
    parser.add_argument("--dervish-retraction-steps", type=int, default=2)
    parser.add_argument("--no-dervish-tangent", action="store_true")
    parser.add_argument("--dervish-head", action="store_true", help="treat lm_head.weight as a frame too")
    parser.add_argument("--no-head-views", action="store_true", help="view c_attn as one matrix instead of per-head frames")
    parser.add_argument("--ns-dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--lr-decay-to", type=float, default=1.0, help="linear decay of every learning rate to this fraction of its start value at the last step (1.0 = constant)")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-mode", default="full", choices=["full", "sampled"])
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--copy-train-marker", default="", help="upweight the loss at the character after this marker")
    parser.add_argument("--copy-loss-weight", type=float, default=1.0)
    parser.add_argument("--copy-probe-marker", default="")
    parser.add_argument("--copy-probe-key-marker", default="key=")
    parser.add_argument("--copy-max-cases", type=int, default=1024)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, default=None, help="append the report as one JSON line")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--quiet", action="store_true")
    return parser


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def resolve_ns_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if device == "cuda" else torch.float32


def mona_beta(args: argparse.Namespace) -> float | None:
    """The MONA beta_a for the hidden-matrix group, or None when MONA is off."""
    beta = getattr(args, "mona_beta", -1.0)
    return None if beta is None or beta < 0 else float(beta)


def build_optimizer(model: TinyTransformer, args: argparse.Namespace, ns_dtype: torch.dtype) -> tuple[torch.optim.Optimizer, dict[str, Any], dict[str, FrameView]]:
    named = list(model.named_parameters())
    views: dict[str, FrameView] = {} if args.no_head_views else model.head_views()
    if args.optimizer == "adamw":
        params = [p for _n, p in named if p.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        return opt, {"optimizer": "adamw", "adam_parameters": sum(p.numel() for p in params), "frame_parameters": 0}, views
    adam_kwargs = {"lr": args.lr, "betas": (args.adam_beta1, args.adam_beta2), "eps": args.adam_eps, "weight_decay": args.weight_decay}
    if args.optimizer == "muon":
        opt, report = build_muon(
            named,
            is_muon=is_hidden,
            muon_kwargs={
                "lr": args.muon_lr,
                "momentum": args.muon_momentum,
                "ns_steps": args.muon_ns_steps,
                "nesterov": not args.no_nesterov,
                "weight_decay": args.weight_decay,
                "ns_dtype": ns_dtype,
                "mona_beta": mona_beta(args),
                "mona_alpha": getattr(args, "mona_alpha", None),
            },
            adam_kwargs=adam_kwargs,
        )
        report["frame_parameters"] = report.pop("muon_parameters")
        return opt, report, views
    predicate = is_hidden_or_head if args.dervish_head else is_hidden
    cap = None if args.dervish_cap is None or args.dervish_cap <= 0 else float(args.dervish_cap)
    opt, report = build_dervish(
        named,
        is_frame=predicate,
        views=views,
        frame_kwargs={
            "lr": args.dervish_lr,
            "momentum": args.muon_momentum,
            "ns_steps": args.muon_ns_steps,
            "nesterov": not args.no_nesterov,
            "ns_dtype": ns_dtype,
            "retraction_steps": args.dervish_retraction_steps,
            "tangent": not args.no_dervish_tangent,
            "scale_lr": args.dervish_scale_lr,
            "scale_cap": cap,
            "scale_floor": args.dervish_floor,
            "scale_decay": args.dervish_scale_decay,
            "mona_beta": mona_beta(args),
            "mona_alpha": getattr(args, "mona_alpha", None),
        },
        adam_kwargs=adam_kwargs,
    )
    opt.initialize()
    return opt, report, views


@torch.no_grad()
def evaluate(
    model: TinyTransformer,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    mode: str,
    batches: int,
    seed: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    stats = AttentionStats()
    model.set_stats(stats)
    total_nll = 0.0
    total_tokens = 0
    logit_max = 0.0
    logit_sq_sum = 0.0
    logit_count = 0
    if mode == "full":
        x_all, y_all = chunked_windows(data, block_size)
        chunks = [(x_all[i : i + batch_size], y_all[i : i + batch_size]) for i in range(0, len(x_all), batch_size)]
    else:
        generator = torch.Generator(device=device).manual_seed(seed + 100_003)
        chunks = [get_batch(data, block_size, batch_size, generator, device) for _ in range(batches)]
    for x, y in chunks:
        logits, loss = model(x, y)
        tokens = y.numel()
        total_nll += float(loss) * tokens
        total_tokens += tokens
        logit_max = max(logit_max, float(logits.abs().max()))
        logit_sq_sum += float(logits.square().sum())
        logit_count += logits.numel()
    model.set_stats(None)
    model.train()
    val_nll = total_nll / max(total_tokens, 1)
    return {
        "val_nll": val_nll,
        "bits_per_char": val_nll / LN2,
        "att_logit_max": max(stats.logit_max) if stats.logit_max else float("nan"),
        "att_entropy_mean": sum(stats.entropy_mean) / len(stats.entropy_mean) if stats.entropy_mean else float("nan"),
        "out_logit_max": logit_max,
        "out_logit_rms": math.sqrt(logit_sq_sum / max(logit_count, 1)),
        "eval_tokens": total_tokens,
    }


@torch.no_grad()
def copy_probe(
    model: TinyTransformer,
    val_text: str,
    codec,
    block_size: int,
    device: str,
    marker: str,
    key_marker: str,
    max_cases: int,
) -> dict[str, Any]:
    """Accuracy and NLL of the character after `marker`, given the line prefix.

    Mirrors the case selection of a reference `copy_answer_probe`:
    every validation line with both markers, prefix cut to the last block_size
    characters, target read from the line (the corpus generator guarantees it
    equals the key character).
    """
    if not marker:
        return {}
    model.eval()
    cases = correct = 0
    total_nll = 0.0
    chance_candidates: set[str] = set()
    for line in val_text.splitlines():
        if cases >= max_cases:
            break
        key_pos = line.find(key_marker)
        marker_pos = line.find(marker)
        if key_pos < 0 or marker_pos < 0:
            continue
        key_at = key_pos + len(key_marker)
        target_at = marker_pos + len(marker)
        if key_at >= len(line) or target_at >= len(line):
            continue
        key = line[key_at]
        target = line[target_at]
        prefix = line[:target_at]
        if target not in codec.stoi or any(ch not in codec.stoi for ch in prefix):
            continue
        chance_candidates.add(key)
        context = prefix[-block_size:]
        ids = torch.tensor([codec.encode(context)], dtype=torch.long, device=device)
        logits, _ = model(ids)
        logprobs = torch.log_softmax(logits[0, -1].float(), dim=-1)
        target_id = codec.stoi[target]
        correct += int(int(logprobs.argmax()) == target_id)
        total_nll += -float(logprobs[target_id])
        cases += 1
    model.train()
    if cases == 0:
        return {"copy_probe_cases": 0, "copy_probe_accuracy": None, "copy_probe_nll": None, "copy_probe_chance": None}
    return {
        "copy_probe_cases": cases,
        "copy_probe_accuracy": correct / cases,
        "copy_probe_nll": total_nll / cases,
        "copy_probe_chance": 1.0 / max(len(chance_candidates), 1),
    }


def answer_position_mask(x: torch.Tensor, marker_ids: list[int]) -> torch.Tensor:
    """True at input position t when x[t-L+1 .. t] spells the marker, so the target at t is the answer."""
    batch, tokens = x.shape
    length = len(marker_ids)
    mask = torch.zeros(batch, tokens, dtype=torch.bool, device=x.device)
    if length == 0 or tokens < length:
        return mask
    hit = torch.ones(batch, tokens - length + 1, dtype=torch.bool, device=x.device)
    for k, token_id in enumerate(marker_ids):
        hit &= x[:, k : tokens - length + 1 + k] == token_id
    mask[:, length - 1 :] = hit
    return mask


def weighted_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    per_token = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none").view_as(y)
    weights = torch.ones_like(per_token) + (weight - 1.0) * mask.to(per_token.dtype)
    return (per_token * weights).sum() / weights.sum()


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    ns_dtype = resolve_ns_dtype(args.ns_dtype, device)
    text = load_corpus(args.corpus, normalize=args.normalize)
    codec = build_codec(text)
    ids = codec.encode(text)
    split_at = split_at_index(len(ids), args.val_fraction, args.block_size)
    train_data = torch.tensor(ids[:split_at], dtype=torch.long, device=device)
    val_data = torch.tensor(ids[split_at:], dtype=torch.long, device=device)
    val_text = text[split_at:]

    torch.manual_seed(args.seed)
    model = TinyTransformer(
        vocab_size=codec.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        pos_encoding=args.pos_encoding,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer, opt_report, views = build_optimizer(model, args, ns_dtype)
    hidden_predicate = is_hidden_or_head if (args.optimizer == "dervish" and args.dervish_head) else is_hidden

    generator = torch.Generator(device=device).manual_seed(args.seed)
    history: list[dict[str, Any]] = []
    train_seconds = 0.0
    t_start = time.perf_counter()
    recent_losses: list[float] = []
    diverged_at: int | None = None

    def record(step: int, train_loss: float | None) -> None:
        ev = evaluate(model, val_data, args.block_size, args.batch_size, args.eval_mode, args.eval_batches, args.seed, device)
        hidden = hidden_matrix_report(model, hidden_predicate, views)
        hidden.pop("per_parameter", None)
        row: dict[str, Any] = {"step": step, "train_loss": train_loss, "elapsed_seconds": time.perf_counter() - t_start, "train_seconds": train_seconds}
        row.update(ev)
        row.update(hidden)
        if args.optimizer == "dervish":
            frames = optimizer.frame_report()
            row["dervish_scale_min"] = min(f["scale_min"] for f in frames)
            row["dervish_scale_max"] = max(f["scale_max"] for f in frames)
            row["dervish_scale_mean"] = sum(f["scale_mean"] for f in frames) / len(frames)
            row["dervish_orthonormality_error_max"] = max(f["orthonormality_error_max"] for f in frames)
            cap = args.dervish_cap if args.dervish_cap and args.dervish_cap > 0 else row["dervish_scale_max"]
            row["att_logit_bound"] = model.attention_logit_bound(cap, cap) if not args.no_head_views else float("nan")
        if args.copy_probe_marker:
            row.update(copy_probe(model, val_text, codec, args.block_size, device, args.copy_probe_marker, args.copy_probe_key_marker, args.copy_max_cases))
        history.append(row)
        if not args.quiet:
            extra = f" copy_acc={row['copy_probe_accuracy']:.3f}" if row.get("copy_probe_accuracy") is not None else ""
            print(
                f"[{args.label or args.optimizer} seed {args.seed}] step {step:5d} train={train_loss if train_loss is not None else float('nan'):.4f} "
                f"val_nll={row['val_nll']:.4f} att_max={row['att_logit_max']:.2f} out_max={row['out_logit_max']:.2f} "
                f"spec_max={row.get('hidden_spectral_norm_max', float('nan')):.3f} srank={row.get('hidden_stable_rank_mean', float('nan')):.2f}{extra}",
                flush=True,
            )

    marker_ids = codec.encode(args.copy_train_marker) if args.copy_train_marker else []
    use_copy_weight = bool(marker_ids) and args.copy_loss_weight != 1.0
    lr_keys = ("lr", "scale_lr")
    base_lrs = [{k: float(g[k]) for k in lr_keys if k in g} for g in optimizer.param_groups]

    def apply_schedule(step: int) -> None:
        if args.lr_decay_to >= 1.0:
            return
        factor = 1.0 - (1.0 - args.lr_decay_to) * (step - 1) / max(args.steps - 1, 1)
        for group, base in zip(optimizer.param_groups, base_lrs):
            for k, v in base.items():
                group[k] = v * factor

    record(0, None)
    model.train()
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        x, y = get_batch(train_data, args.block_size, args.batch_size, generator, device)
        if use_copy_weight:
            logits, _ = model(x)
            loss = weighted_loss(logits, y, answer_position_mask(x, marker_ids), args.copy_loss_weight)
        else:
            _logits, loss = model(x, y)
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            diverged_at = step
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        apply_schedule(step)
        optimizer.step()
        train_seconds += time.perf_counter() - t0
        recent_losses.append(loss_value)
        if len(recent_losses) > 20:
            recent_losses.pop(0)
        if not args.quiet and args.log_every > 0 and step % args.log_every == 0 and step % args.eval_interval != 0:
            print(f"[{args.label or args.optimizer} seed {args.seed}] step {step:5d} train={loss_value:.4f}", flush=True)
        if step % args.eval_interval == 0 or step == args.steps:
            record(step, sum(recent_losses) / len(recent_losses))

    final = history[-1]
    peak_att = max(h["att_logit_max"] for h in history if math.isfinite(h["att_logit_max"]))
    first_copy_step = None
    if args.copy_probe_marker:
        for h in history:
            acc = h.get("copy_probe_accuracy")
            if acc is not None and acc >= 0.25:
                first_copy_step = h["step"]
                break
    report: dict[str, Any] = {
        "label": args.label or args.optimizer,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "corpus": args.corpus.name,
        "corpus_chars": len(text),
        "vocab_size": codec.vocab_size,
        "steps": args.steps,
        "steps_completed": diverged_at - 1 if diverged_at else args.steps,
        "diverged_at": diverged_at,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "pos_encoding": args.pos_encoding,
        "lr": args.lr,
        "lr_decay_to": args.lr_decay_to,
        "weight_decay": args.weight_decay,
        "muon_lr": args.muon_lr if args.optimizer == "muon" else None,
        "dervish_lr": args.dervish_lr if args.optimizer == "dervish" else None,
        "mona_beta": mona_beta(args) if args.optimizer != "adamw" else None,
        "mona_alpha": getattr(args, "mona_alpha", None) if args.optimizer != "adamw" and mona_beta(args) is not None else None,
        "dervish_scale_lr": args.dervish_scale_lr if args.optimizer == "dervish" else None,
        "dervish_cap": (args.dervish_cap if args.dervish_cap and args.dervish_cap > 0 else None) if args.optimizer == "dervish" else None,
        "dervish_floor": args.dervish_floor if args.optimizer == "dervish" else None,
        "dervish_head": bool(args.dervish_head) if args.optimizer == "dervish" else None,
        "head_views": not args.no_head_views,
        "copy_train_marker": args.copy_train_marker or None,
        "copy_loss_weight": args.copy_loss_weight if use_copy_weight else None,
        "ns_dtype": str(ns_dtype).replace("torch.", ""),
        "trainable_params": trainable,
        "frame_params": opt_report.get("frame_parameters", 0),
        "adam_params": opt_report.get("adam_parameters", 0),
        "eval_mode": args.eval_mode,
        "val_nll": final["val_nll"],
        "bits_per_char": final["bits_per_char"],
        "best_val_nll": min(h["val_nll"] for h in history),
        "train_loss_final": final["train_loss"],
        "att_logit_max_final": final["att_logit_max"],
        "att_logit_max_peak": peak_att,
        "att_entropy_mean_final": final["att_entropy_mean"],
        "out_logit_max_final": final["out_logit_max"],
        "out_logit_max_peak": max(h["out_logit_max"] for h in history),
        "hidden_spectral_norm_max_final": final.get("hidden_spectral_norm_max"),
        "hidden_spectral_norm_max_peak": max(h.get("hidden_spectral_norm_max", 0.0) for h in history),
        "hidden_stable_rank_mean_final": final.get("hidden_stable_rank_mean"),
        "hidden_effective_rank_mean_final": final.get("hidden_effective_rank_mean"),
        "hidden_sigma_min_min_final": final.get("hidden_sigma_min_min"),
        "dervish_orthonormality_error_max_final": final.get("dervish_orthonormality_error_max"),
        "att_logit_bound_final": final.get("att_logit_bound"),
        "copy_probe_cases": final.get("copy_probe_cases"),
        "copy_probe_accuracy": final.get("copy_probe_accuracy"),
        "copy_probe_nll": final.get("copy_probe_nll"),
        "copy_probe_chance": final.get("copy_probe_chance"),
        "copy_probe_first_step_at_or_above_025": first_copy_step,
        "train_seconds": train_seconds,
        "total_seconds": time.perf_counter() - t_start,
        "device": device,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "history": history,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report) + "\n")
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = train(args)
    summary = {k: v for k, v in report.items() if k != "history"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
