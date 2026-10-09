"""Comparison matrix: named configs x seeds -> runs/<stage>.jsonl + runs/<stage>.md.

Mirrors the shape of a reference comparison runner: one JSON report per line, a
markdown summary with per-config means over seeds, then the raw rows.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.train import build_parser as build_train_parser  # noqa: E402
from lab.train import train  # noqa: E402

# Each config overrides the matrix-level base arguments.
CONFIGS: dict[str, dict[str, Any]] = {
    "adamw": {"optimizer": "adamw"},
    "muon": {"optimizer": "muon"},
    "muon_wd": {"optimizer": "muon", "weight_decay": 0.1},
    "dervish": {"optimizer": "dervish"},
    "dervish_nocap": {"optimizer": "dervish", "dervish_cap": 0.0},
    "dervish_head": {"optimizer": "dervish", "dervish_head": True},
    "dervish_flat": {"optimizer": "dervish", "no_head_views": True},
    "dervish_notangent": {"optimizer": "dervish", "no_dervish_tangent": True},
    # Copy-position weighting (a behaviour-formation lever)
    "adamw_copyw": {"optimizer": "adamw", "copy_train_marker": "answer=", "copy_loss_weight": 50.0},
    "muon_copyw": {"optimizer": "muon", "copy_train_marker": "answer=", "copy_loss_weight": 50.0},
    "muon_wd_copyw": {"optimizer": "muon", "weight_decay": 0.1, "copy_train_marker": "answer=", "copy_loss_weight": 50.0},
    "dervish_copyw": {"optimizer": "dervish", "copy_train_marker": "answer=", "copy_loss_weight": 50.0},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dervish lab comparison matrix", parents=[build_train_parser()], conflict_handler="resolve")
    parser.add_argument("--configs", nargs="+", default=["adamw", "muon", "dervish"], choices=sorted(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 19])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--title", default="Dervish Lab Summary")
    parser.add_argument("--sweep-key", default="", help="optional: a train argument to sweep (e.g. muon_lr)")
    parser.add_argument("--sweep-values", nargs="*", type=float, default=[])
    parser.add_argument("--sweep-configs", nargs="*", default=[], help="configs the sweep applies to (default: all)")
    return parser


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "·"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None and not (isinstance(r[key], float) and math.isnan(r[key]))]
    return statistics.fmean(values) if values else None


def summarize(rows: list[dict[str, Any]], title: str, command: str) -> str:
    lines = [f"# {title}", "", "Command:", "", "```", command, "```", ""]
    has_copy = any(r.get("copy_probe_accuracy") is not None for r in rows)
    swept = any("sweep_value" in r for r in rows)
    groups: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["label"], r.get("sweep_value")), []).append(r)
    header = ["Config"]
    if swept:
        header.append("Sweep")
    header += ["Seeds", "Trainable params", "Mean val NLL", "Min val NLL", "Max val NLL", "Mean bits/char", "Peak att logit (mean)", "Final spectral max (mean)", "Final stable rank (mean)", "Final out logit max (mean)"]
    if has_copy:
        header += ["Mean copy acc", "Min copy acc", "Max copy acc", "Mean copy NLL"]
    header += ["Diverged", "Mean train s"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([" --- "] + [" ---: "] * (len(header) - 1)) + "|")
    for (label, sweep_value), group in groups.items():
        vals = [r["val_nll"] for r in group]
        cells = [label]
        if swept:
            cells.append(fmt(sweep_value, 4))
        cells += [
            str(len(group)),
            str(group[0]["trainable_params"]),
            fmt(statistics.fmean(vals)),
            fmt(min(vals)),
            fmt(max(vals)),
            fmt(mean_of(group, "bits_per_char")),
            fmt(mean_of(group, "att_logit_max_peak"), 2),
            fmt(mean_of(group, "hidden_spectral_norm_max_final"), 3),
            fmt(mean_of(group, "hidden_stable_rank_mean_final"), 2),
            fmt(mean_of(group, "out_logit_max_final"), 2),
        ]
        if has_copy:
            accs = [r["copy_probe_accuracy"] for r in group if r.get("copy_probe_accuracy") is not None]
            cells += [fmt(statistics.fmean(accs) if accs else None), fmt(min(accs) if accs else None), fmt(max(accs) if accs else None), fmt(mean_of(group, "copy_probe_nll"))]
        cells += [str(sum(1 for r in group if r.get("diverged_at"))), fmt(mean_of(group, "train_seconds"), 1)]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Raw rows:", ""]
    raw_header = ["Config", "Seed"] + (["Sweep"] if swept else []) + ["val NLL", "bits/char", "peak att logit", "final att logit", "final spectral max", "final stable rank", "final out logit max"] + (["copy acc", "copy NLL"] if has_copy else []) + ["diverged at", "train s"]
    lines.append("| " + " | ".join(raw_header) + " |")
    lines.append("|" + "|".join([" --- ", " ---: "] + [" ---: "] * (len(raw_header) - 2)) + "|")
    for r in rows:
        cells = [r["label"], str(r["seed"])] + ([fmt(r.get("sweep_value"), 4)] if swept else []) + [
            fmt(r["val_nll"]), fmt(r["bits_per_char"]), fmt(r["att_logit_max_peak"], 2), fmt(r["att_logit_max_final"], 2),
            fmt(r.get("hidden_spectral_norm_max_final"), 3), fmt(r.get("hidden_stable_rank_mean_final"), 2), fmt(r.get("out_logit_max_final"), 2),
        ]
        if has_copy:
            cells += [fmt(r.get("copy_probe_accuracy")), fmt(r.get("copy_probe_nll"))]
        cells += [fmt(r.get("diverged_at")), fmt(r.get("train_seconds"), 1)]
        lines.append("| " + " | ".join(cells) + " |")
    # per-eval curves (means over seeds) for the main metrics
    lines += ["", "Evaluation curves (means over seeds):", ""]
    for (label, sweep_value), group in groups.items():
        steps = [h["step"] for h in group[0]["history"]]
        if not all([h["step"] for h in r["history"]] == steps for r in group):
            continue
        name = label if not swept else f"{label} @ {fmt(sweep_value, 4)}"
        lines.append(f"`{name}`")
        lines.append("")
        cols = ["step", "val NLL", "att logit max", "out logit max", "spectral max", "stable rank"] + (["copy acc"] if has_copy else [])
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join([" ---: "] * len(cols)) + "|")
        for i, step in enumerate(steps):
            hs = [r["history"][i] for r in group]
            row = [str(step), fmt(mean_of(hs, "val_nll"), 4), fmt(mean_of(hs, "att_logit_max"), 2), fmt(mean_of(hs, "out_logit_max"), 2), fmt(mean_of(hs, "hidden_spectral_norm_max"), 3), fmt(mean_of(hs, "hidden_stable_rank_mean"), 2)]
            if has_copy:
                row.append(fmt(mean_of(hs, "copy_probe_accuracy"), 4))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = "python " + " ".join(shlex.quote(a) for a in sys.argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    rows: list[dict[str, Any]] = []
    sweep_values: list[float | None] = list(args.sweep_values) if args.sweep_key else [None]
    sweep_targets = set(args.sweep_configs) if args.sweep_configs else set(args.configs)
    matrix_start = time.perf_counter()
    for config_name in args.configs:
        for sweep_value in sweep_values:
            if sweep_value is not None and config_name not in sweep_targets:
                continue
            for seed in args.seeds:
                run_args = argparse.Namespace(**vars(args))
                for key, value in CONFIGS[config_name].items():
                    setattr(run_args, key, value)
                if sweep_value is not None:
                    setattr(run_args, args.sweep_key, sweep_value)
                run_args.seed = seed
                run_args.label = config_name
                run_args.out = None
                report = train(run_args)
                if sweep_value is not None:
                    report["sweep_key"] = args.sweep_key
                    report["sweep_value"] = sweep_value
                rows.append(report)
                with args.out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(report) + "\n")
                print(f"== {config_name} seed {seed}" + (f" {args.sweep_key}={sweep_value}" if sweep_value is not None else "") + f": val_nll={report['val_nll']:.6f} peak_att={report['att_logit_max_peak']:.2f} train_s={report['train_seconds']:.1f}", flush=True)
    summary = summarize(rows, args.title, command)
    summary += f"\nMatrix wall-clock: {time.perf_counter() - matrix_start:.1f} s · platform: {rows[0]['platform']} · torch {rows[0]['torch_version']} · device {rows[0]['device']} · threads {rows[0]['torch_threads']}\n"
    args.summary.write_text(summary, encoding="utf-8")
    print(f"wrote {args.out} and {args.summary}")


if __name__ == "__main__":
    main()
