"""Compute the verdicts of docs/HYPOTHESIS-D001.md from the raw run files.

The pass, partial and kill lines were fixed before the runs; this script applies
them mechanically so the verdict is computed, not eyeballed. It prints every
number it uses. Exit code 0 always; the verdicts are the output.

Usage: python lab/check_claims.py [--runs runs]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def by_label(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["label"], []).append(r)
    return out


def claim_1(e1: dict[str, list[dict]]) -> str:
    print("Claim 1 · bounded spectra and logits (E1, every evaluation of every dervish seed)")
    dervish = e1.get("dervish", [])
    muon = e1.get("muon", [])
    if not dervish or not muon:
        print("  missing arms; cannot decide")
        return "UNDECIDED"
    spectral_ok = True
    logit_ok = True
    worst_spec = 0.0
    worst_margin = math.inf
    for r in dervish:
        for h in r["history"]:
            worst_spec = max(worst_spec, h["hidden_spectral_norm_max"])
            if h["hidden_spectral_norm_max"] > 1.0 + 1e-4:
                spectral_ok = False
            bound = h.get("att_logit_bound")
            if bound is None or math.isnan(bound):
                logit_ok = False
                continue
            worst_margin = min(worst_margin, bound - h["att_logit_max"])
            if h["att_logit_max"] > bound:
                logit_ok = False
    print(f"  dervish: max hidden spectral norm over all evaluations {worst_spec:.6f} (cap 1.0) -> {'ok' if spectral_ok else 'VIOLATED'}")
    print(f"  dervish: smallest (bound - measured attention logit) over all evaluations {worst_margin:.4f} -> {'ok' if logit_ok else 'VIOLATED'}")
    growth = []
    for r in muon:
        s0 = r["history"][0]["hidden_spectral_norm_max"]
        s1 = r["history"][-1]["hidden_spectral_norm_max"]
        growth.append(s1 / s0)
        print(f"  muon seed {r['seed']}: spectral max step 0 {s0:.4f} -> step {r['history'][-1]['step']} {s1:.4f} (x{s1 / s0:.2f}); peak attention logit {r['att_logit_max_peak']:.2f}")
    mean_growth = statistics.fmean(growth)
    print(f"  muon mean growth x{mean_growth:.2f} (pass line: >= 2x)")
    if not (spectral_ok and logit_ok):
        verdict = "KILL"
    elif mean_growth >= 2.0:
        verdict = "PASS"
    else:
        verdict = "PARTIAL"
    print(f"  verdict: {verdict}")
    return verdict


def claim_2(e1: dict[str, list[dict]]) -> str:
    print("Claim 2 · quality parity (E1, mean final val NLL over seeds; dervish vs best of muon, muon_wd)")
    needed = ["dervish", "muon", "muon_wd"]
    if any(k not in e1 for k in needed):
        print("  missing arms; cannot decide")
        return "UNDECIDED"
    means = {k: statistics.fmean(r["val_nll"] for r in e1[k]) for k in e1}
    for k, v in sorted(means.items(), key=lambda kv: kv[1]):
        print(f"  {k:<14} mean val NLL {v:.6f}  (seeds {[r['seed'] for r in e1[k]]}, values {[round(r['val_nll'], 6) for r in e1[k]]})")
    best_muon = min(means["muon"], means["muon_wd"])
    delta = means["dervish"] - best_muon
    print(f"  dervish - best muon arm = {delta:+.6f}  (pass <= +0.030, partial <= +0.100)")
    verdict = "PASS" if delta <= 0.030 else "PARTIAL" if delta <= 0.100 else "KILL"
    print(f"  verdict: {verdict}")
    return verdict


def basin(rows: list[dict]) -> tuple[int, float, list[tuple[float, float, bool]]]:
    pts = sorted(rows, key=lambda r: r["sweep_value"])
    finite = [r["val_nll"] for r in pts if not r.get("diverged_at") and math.isfinite(r["val_nll"]) and r["val_nll"] <= 3.0]
    best = min(finite) if finite else math.inf
    table = []
    count = 0
    for r in pts:
        inside = (not r.get("diverged_at")) and math.isfinite(r["val_nll"]) and r["val_nll"] <= 3.0 and r["val_nll"] <= best + 0.100
        count += int(inside)
        table.append((r["sweep_value"], r["val_nll"], inside))
    return count, best, table


def claim_3(e2: dict[str, list[dict]]) -> str:
    print("Claim 3 · learning-rate robustness (E2, seed 7, 600 steps; basin = grid points within 0.100 of the optimizer's best)")
    if not e2.get("muon") or not e2.get("dervish"):
        print("  missing sweeps; cannot decide")
        return "UNDECIDED"
    sizes = {}
    for name in ("muon", "dervish", "adamw"):
        if not e2.get(name):
            continue
        count, best, table = basin(e2[name])
        sizes[name] = count
        cells = ", ".join(f"{lr:g}: {nll:.4f}{'' if inside else ' (out)'}" for lr, nll, inside in table)
        print(f"  {name:<8} best {best:.4f}  basin {count}/{len(table)}  [{cells}]")
    verdict = "PASS" if sizes["dervish"] >= sizes["muon"] + 1 else "PARTIAL" if sizes["dervish"] == sizes["muon"] else "KILL"
    print(f"  verdict: {verdict}")
    return verdict


def exploratory(e1: dict[str, list[dict]], e3: dict[str, list[dict]]) -> None:
    print("Exploratory · stable rank (E1 final, mean over seeds)")
    for k, rows in e1.items():
        print(f"  {k:<14} stable rank {statistics.fmean(r['hidden_stable_rank_mean_final'] for r in rows):.2f}  effective rank {statistics.fmean(r['hidden_effective_rank_mean_final'] for r in rows):.2f}")
    if e3:
        print("Exploratory · copy behaviour (E3): first eval step with copy accuracy >= 0.25 per seed, final accuracy mean")
        for k, rows in e3.items():
            firsts = [r.get("copy_probe_first_step_at_or_above_025") for r in rows]
            accs = [r.get("copy_probe_accuracy") for r in rows if r.get("copy_probe_accuracy") is not None]
            print(f"  {k:<16} first>=0.25 at {firsts}  final acc mean {statistics.fmean(accs) if accs else float('nan'):.4f}  per seed {[round(a, 4) for a in accs]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    e1 = by_label(load(args.runs / "e1_text.jsonl"))
    e2 = {name: load(args.runs / f"e2_{name}.jsonl") for name in ("muon", "dervish", "adamw")}
    e3 = by_label(load(args.runs / "e3_copy.jsonl"))
    v1 = claim_1(e1)
    print()
    v2 = claim_2(e1)
    print()
    v3 = claim_3(e2)
    print()
    exploratory(e1, e3)
    print()
    print(f"VERDICTS  claim 1: {v1}   claim 2: {v2}   claim 3: {v3}")


if __name__ == "__main__":
    main()
