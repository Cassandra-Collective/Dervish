"""Figures from the raw run files. Every number drawn here comes from runs/*.jsonl.

Usage: python lab/figures.py [--runs runs] [--out figures]
Produces PNGs only for the matrices whose jsonl exists, so it can be run on a
partial set.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (dataviz skill reference instance, light surface),
# assigned to entities in a fixed order across every figure.
SERIES = {
    "dervish": ("#2a78d6", "Dervish (cap 1.0)"),
    "muon": ("#eb6834", "Muon, no decay"),
    "muon_wd": ("#1baf7a", "Muon, weight decay 0.1"),
    "adamw": ("#eda100", "AdamW"),
    "dervish_nocap": ("#e87ba4", "Dervish, no cap"),
}
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=10)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def by_label(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["label"], []).append(r)
    return out


def series_style(label: str) -> tuple[str, str, str]:
    base = label.replace("_copyw", "")
    color, name = SERIES.get(base, ("#52514e", label))
    dash = "--" if label.endswith("_copyw") else "-"
    if label.endswith("_copyw"):
        name += ", copy weight 50"
    return color, name, dash


def curve(ax, group: list[dict], key: str, label: str, transform=None) -> None:
    steps = [h["step"] for h in group[0]["history"]]
    values = []
    for i in range(len(steps)):
        vals = [r["history"][i].get(key) for r in group if i < len(r["history"])]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if transform:
            vals = [transform(v) for v in vals]
        values.append(vals)
    mean = [statistics.fmean(v) if v else float("nan") for v in values]
    lo = [min(v) if v else float("nan") for v in values]
    hi = [max(v) if v else float("nan") for v in values]
    color, name, dash = series_style(label)
    ax.fill_between(steps, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.plot(steps, mean, color=color, linewidth=2, linestyle=dash, solid_capstyle="round", label=f"{name} (n={len(group)})")
    ax.annotate(f"{mean[-1]:.3g}", (steps[-1], mean[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=TEXT_2, va="center")


def fig_curves(rows: list[dict], out: Path, stem: str, title_prefix: str) -> None:
    groups = by_label(rows)
    panels = [
        ("val_nll", "Validation NLL (nats per character)", None),
        ("att_logit_max", "Largest attention logit on the validation set", None),
        ("hidden_spectral_norm_max", "Largest hidden spectral norm", None),
        ("hidden_stable_rank_mean", "Mean stable rank of hidden matrices", None),
    ]
    if any(r.get("copy_probe_accuracy") is not None for r in rows):
        panels.append(("copy_probe_accuracy", "Copy probe accuracy (chance 0.125)", None))
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.5, 3.2 * len(panels)), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, ylabel, transform) in zip(axes, panels):
        for label in sorted(groups, key=lambda l: list(SERIES).index(l.replace("_copyw", "")) if l.replace("_copyw", "") in SERIES else 99):
            curve(ax, groups[label], key, label, transform)
        style(ax, f"{title_prefix}: {ylabel}", "training step", ylabel.split(" (")[0])
        if key == "att_logit_max" and "dervish" in groups:
            bound = [h.get("att_logit_bound") for h in groups["dervish"][0]["history"]]
            if all(b is not None and not math.isnan(b) for b in bound):
                steps = [h["step"] for h in groups["dervish"][0]["history"]]
                ax.plot(steps, bound, color=SERIES["dervish"][0], linewidth=1, linestyle=":", label="Dervish analytic bound (Proposition 2)")
        if key == "hidden_spectral_norm_max":
            ax.axhline(1.0, color=SERIES["dervish"][0], linewidth=1, linestyle=":", label="Dervish cap")
        if key == "copy_probe_accuracy":
            ax.axhline(0.125, color=TEXT_2, linewidth=1, linestyle=":", label="chance")
        ax.legend(fontsize=7.5, frameon=False, loc="best", labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", facecolor=SURFACE)
    plt.close(fig)


def fig_lr_basins(sweeps: dict[str, list[dict]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    best_overall = None
    for label, rows in sweeps.items():
        if not rows:
            continue
        color, name, _ = series_style(label)
        pts = sorted(rows, key=lambda r: r["sweep_value"])
        xs = [r["sweep_value"] for r in pts]
        ys = [r["val_nll"] if not r.get("diverged_at") and math.isfinite(r["val_nll"]) else float("nan") for r in pts]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=f"{name}: final NLL vs {pts[0]['sweep_key']}")
        for r, x, y in zip(pts, xs, ys):
            if r.get("diverged_at") or not math.isfinite(r["val_nll"]) or r["val_nll"] > 3.0:
                ax.plot([x], [min(3.0, y if math.isfinite(y) else 3.0)], color=color, marker="x", markersize=9, linestyle="none")
                ax.annotate("diverged" if r.get("diverged_at") else "> 3.0", (x, 3.0), xytext=(0, 6), textcoords="offset points", fontsize=7, color=TEXT_2, ha="center")
        finite = [y for y in ys if math.isfinite(y)]
        if finite:
            best_overall = min(finite) if best_overall is None else min(best_overall, min(finite))
    ax.set_xscale("log", base=2)
    ax.set_ylim(top=3.05)
    style(ax, "E2: learning-rate basins, seed 7, 600 steps (lower is better; x marks a run outside the basin rule)", "learning rate of the swept group (log scale)", "final validation NLL")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(out / "e2_lr_basins.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    e1 = load(args.runs / "e1_text.jsonl")
    if e1:
        fig_curves(e1, args.out, "e1_text_curves", "E1 Tiny Shakespeare")
        print("wrote e1_text_curves.png")
    e3 = load(args.runs / "e3_copy.jsonl")
    plain = [r for r in e3 if not r["label"].endswith("_copyw")]
    weighted = [r for r in e3 if r["label"].endswith("_copyw")]
    if plain:
        fig_curves(plain, args.out, "e3_copy_plain_curves", "E3 copy corpus, plain training")
        print("wrote e3_copy_plain_curves.png")
    if weighted:
        fig_curves(weighted, args.out, "e3_copy_weighted_curves", "E3 copy corpus, copy weight 50")
        print("wrote e3_copy_weighted_curves.png")
    sweeps = {name: load(args.runs / f"e2_{name}.jsonl") for name in ("muon", "dervish", "adamw")}
    if any(sweeps.values()):
        fig_lr_basins(sweeps, args.out)
        print("wrote e2_lr_basins.png")


if __name__ == "__main__":
    main()
