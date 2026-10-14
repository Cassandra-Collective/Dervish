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
    dash = "--" if label.endswith("_copyw") or label == "dervish_nocap" else "-"
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


def fig_scale_lr(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for label in ("dervish", "dervish_nocap"):
        pts = sorted((r for r in rows if r["label"] == label), key=lambda r: r["sweep_value"])
        if not pts:
            continue
        color, name, dash = series_style(label)
        ax.plot([r["sweep_value"] for r in pts], [r["val_nll"] for r in pts], color=color, linewidth=2, linestyle=dash, marker="o", markersize=5, label=name)
        for r in pts:
            ax.annotate(f"spec {r['hidden_spectral_norm_max_final']:.2g}", (r["sweep_value"], r["val_nll"]), xytext=(0, 7), textcoords="offset points", fontsize=7, color=TEXT_2, ha="center")
    ax.set_xscale("log")
    style(ax, "E4 (post-hoc): final NLL vs spectrum learning rate, seed 7, 1000 steps (labels: final spectral max)", "spectrum learning rate (log scale)", "final validation NLL")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(out / "e4_scale_lr.png", facecolor=SURFACE)
    plt.close(fig)


def fig_replication(e1: list[dict], e6: list[dict], e7: list[dict], out: Path) -> None:
    """E6/E7 beside Stage D1: three-seed validation NLL curves (mean, min-max band)."""
    groups = by_label([r for r in e1 if r["label"] in ("muon", "muon_wd")]) | by_label(e6) | by_label(e7)
    palette = {"muon": "#eb6834", "muon_wd": "#1baf7a", "dervish_tuned": "#2a78d6", "dervish_tuned_decay": "#4a3aa7", "muon_decay": "#e34948"}
    names = {"muon": "Muon, no decay (D1)", "muon_wd": "Muon, weight decay 0.1 (D1)", "dervish_tuned": "Dervish tuned, cap 1.0 (E6)", "dervish_tuned_decay": "Dervish tuned + linear decay (E7)", "muon_decay": "Muon + the same linear decay (E7)"}
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax, key in zip(axes, ("val_nll", "att_logit_max")):
        for label, group in groups.items():
            steps = [h["step"] for h in group[0]["history"]]
            series = [[r["history"][i][key] for r in group] for i in range(len(steps))]
            mean = [statistics.fmean(v) for v in series]
            ax.fill_between(steps, [min(v) for v in series], [max(v) for v in series], color=palette[label], alpha=0.12, linewidth=0)
            ax.plot(steps, mean, color=palette[label], linewidth=2, linestyle="--" if "decay" in label else "-", label=f"{names[label]} (n={len(group)})")
            ax.annotate(f"{mean[-1]:.3g}", (steps[-1], mean[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color=TEXT_2, va="center")
    style(axes[0], "E6 and E7 (post-hoc, three seeds) beside Stage D1: validation NLL", "training step", "validation NLL")
    axes[0].set_ylim(1.4, 2.6)
    style(axes[1], "E6 and E7 beside Stage D1: largest attention logit", "training step", "attention logit max")
    axes[0].legend(fontsize=7, frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(out / "e6_e7_replication.png", facecolor=SURFACE)
    plt.close(fig)


def fig_tuned(rows: list[dict], out: Path) -> None:
    """E5: validation NLL and attention-logit curves of the tuned arms, one line per run."""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    for i, r in enumerate(rows):
        color = palette[i % len(palette)]
        steps = [h["step"] for h in r["history"]]
        dash = "--" if r["label"].startswith("muon") else "-"
        for ax, key in zip(axes, ("val_nll", "att_logit_max")):
            ys = [h[key] for h in r["history"]]
            ax.plot(steps, ys, color=color, linewidth=2, linestyle=dash, label=r["label"])
            ax.annotate(f"{ys[-1]:.3g}", (steps[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color=TEXT_2, va="center")
    style(axes[0], "E5 (post-hoc, seed 7): validation NLL of the tuned arms", "training step", "validation NLL")
    axes[0].set_ylim(1.4, 2.6)
    style(axes[1], "E5 (post-hoc, seed 7): largest attention logit (dashed = Muon arms)", "training step", "attention logit max")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=7, frameon=False, labelcolor=TEXT, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "e5_tuned_curves.png", facecolor=SURFACE)
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
    style(ax, "E2: learning-rate basins, seed 7, 600 steps (lower is better; Dervish includes the E2b extension)", "learning rate of the swept group (log scale)", "final validation NLL")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(out / "e2_lr_basins.png", facecolor=SURFACE)
    plt.close(fig)


D8_FAMILIES = {  # family -> (colour, dash, display name)
    "Muon": ("#eb6834", "-", "Muon"),
    "MONA": ("#4a3aa7", "-", "MONA"),
    "Dervish": ("#2a78d6", "-", "Dervish (cap 1.0)"),
    "Dervish + MONA": ("#2a78d6", "--", "Dervish + MONA term"),
}


def d8_family(label: str) -> str:
    if label.startswith("dervish_tuned_mona"):
        return "Dervish + MONA"
    if label.startswith("dervish"):
        return "Dervish"
    return "MONA" if label.startswith("mona") else "Muon"


def fig_d8(rows: list[dict], out: Path) -> None:
    """Stage D8b: curves per schedule, and per-seed paired differences at step 1000."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        schedule = "decay" if "_decay" in r["label"] else "constant"
        groups.setdefault((schedule, d8_family(r["label"])), []).append(r)
    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for col, schedule in enumerate(("constant", "decay")):
        for family, (color, dash, name) in D8_FAMILIES.items():
            group = groups.get((schedule, family), [])
            if not group:
                continue
            steps = [h["step"] for h in group[0]["history"]]
            beta = group[0].get("mona_beta")
            lr = group[0].get("muon_lr")
            tag = name + (f", lr {lr:g}" if lr else "") + (f", beta_a {beta:g}" if beta else "")
            for row, key in enumerate(("val_nll", "att_logit_max")):
                series = [[r["history"][i][key] for r in group] for i in range(len(steps))]
                mean = [statistics.fmean(v) for v in series]
                ax = axes[row][col]
                ax.fill_between(steps, [min(v) for v in series], [max(v) for v in series], color=color, alpha=0.10, linewidth=0)
                ax.plot(steps, mean, color=color, linewidth=2, linestyle=dash, label=f"{tag} (n={len(group)})")
                ax.annotate(f"{mean[-1]:.3g}", (steps[-1], mean[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, color=TEXT_2, va="center")
        style(axes[0][col], f"D8b {schedule} rate: validation NLL, fresh seeds", "training step", "validation NLL")
        axes[0][col].set_ylim(1.45, 1.85)
        style(axes[1][col], f"D8b {schedule} rate: largest attention logit", "training step", "attention logit max")
        axes[1][col].set_yscale("log")
        axes[0][col].legend(fontsize=7, frameon=False, labelcolor=TEXT)
        # paired differences at step 1000
        ax = axes[2][col]
        nll = {f: {r["seed"]: r["val_nll"] for r in groups.get((schedule, f), [])} for f in D8_FAMILIES}
        means = {f: statistics.fmean(nll[f].values()) for f in ("Muon", "MONA") if nll[f]}
        ref = min(means, key=means.get) if means else "Muon"
        pairs = [("MONA - Muon", "MONA", "Muon"), (f"Dervish - {ref}", "Dervish", ref), ("Dervish+MONA - Dervish", "Dervish + MONA", "Dervish")]
        for i, (title, a, b) in enumerate(pairs):
            seeds = sorted(set(nll[a]) & set(nll[b]))
            diffs = [nll[a][s] - nll[b][s] for s in seeds]
            if len(diffs) < 2:
                continue
            mean = statistics.fmean(diffs)
            half = {4: 2.776}.get(len(diffs) - 1, 2.0) * statistics.stdev(diffs) / math.sqrt(len(diffs))
            ax.scatter([d for d in diffs], [i] * len(diffs), color=TEXT_2, s=14, zorder=3)
            ax.errorbar([mean], [i], xerr=[[half], [half]], color=D8_FAMILIES[a][0], capsize=4, linewidth=2, marker="o", markersize=6, zorder=4)
            ax.annotate(f"{mean:+.3f}", (mean, i), xytext=(0, 8), textcoords="offset points", fontsize=7, color=TEXT, ha="center")
        ax.axvline(0.0, color=TEXT_2, linewidth=1)
        ax.axvline(0.030, color="#2a78d6", linewidth=1, linestyle=":", label="Q2 parity line +0.030")
        ax.set_yticks(range(len(pairs)), [p[0] for p in pairs])
        style(ax, f"D8b {schedule}: paired differences at step 1000", "NLL difference (dots: seeds; bar: mean, 95% CI; < 0: first arm better)", "")
        ax.legend(fontsize=7, frameon=False, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "d8_mona_comparison.png", facecolor=SURFACE)
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
    sweeps["dervish"] = sweeps["dervish"] + load(args.runs / "e2b_dervish_low.jsonl")  # post-hoc lower grid joins the series
    if any(sweeps.values()):
        fig_lr_basins(sweeps, args.out)
        print("wrote e2_lr_basins.png")
    e4 = load(args.runs / "e4_scale_lr.jsonl")
    if e4:
        fig_scale_lr(e4, args.out)
        print("wrote e4_scale_lr.png")
    e5 = load(args.runs / "e5_tuned.jsonl")
    if e5:
        fig_tuned(e5, args.out)
        print("wrote e5_tuned_curves.png")
    e6 = load(args.runs / "e6_replicate.jsonl")
    e7 = load(args.runs / "e7_decay_pair.jsonl")
    if e1 and (e6 or e7):
        fig_replication(e1, e6, e7, args.out)
        print("wrote e6_e7_replication.png")
    d8 = load(args.runs / "d8" / "d8b_eval.jsonl")
    if d8:
        fig_d8(d8, args.out)
        print("wrote d8_mona_comparison.png")


if __name__ == "__main__":
    main()
