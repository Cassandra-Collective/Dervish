"""Compute the verdicts of docs/HYPOTHESIS-D002.md from the raw run files.

Reads `runs/d8/d8a_selection.jsonl` (seed 7, the selection grid) and
`runs/d8/d8b_eval.jsonl` (fresh seeds), re-derives the selection with the
registered rule, checks this machine against the record, and applies the Q1 to
Q4 lines mechanically. It prints every number it uses. Exit code 0 always; the
verdicts are the output.

Usage: python lab/check_d002.py [--runs runs/d8]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

# two-sided 95% Student t quantiles by degrees of freedom (the registration uses df 4)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
SCHEDULES = {"constant": "", "decay": "_decay"}
FAMILIES = {
    "Muon": ["muon", "muon_lr01"],
    "MONA": ["mona", "mona_lr01"],
    "Dervish": ["dervish_tuned"],
    "Dervish + MONA": ["dervish_tuned_mona"],
}
# seed-7 values in docs/RESULTS.md (Stages D1, D5, D7), for the machine check
RECORD = {
    ("muon", None): 1.538644,
    ("muon_lr01", None): 1.527197,
    ("dervish_tuned", None): 1.590136,
    ("muon_decay", None): 1.491569,
    ("dervish_tuned_decay", None): 1.511704,
}
PARITY, PARTIAL = 0.030, 0.100
CAP = 1.0


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key(row: dict) -> tuple[str, float | None]:
    return row["label"], row.get("mona_beta")


def arm_name(k: tuple[str, float | None]) -> str:
    return k[0] + (f" (beta_a {k[1]})" if k[1] is not None else "")


def paired(a: dict[int, float], b: dict[int, float]) -> tuple[float, float, float, int]:
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    n = len(diffs)
    mean = statistics.fmean(diffs)
    half = T95[n - 1] * statistics.stdev(diffs) / math.sqrt(n) if n > 1 else math.inf
    return mean, mean - half, mean + half, n


def interval_verdict(lo: float, hi: float, below: str, contains: str, above: str) -> str:
    if hi < 0:
        return below
    if lo > 0:
        return above
    return contains


def select(rows: list[dict]) -> dict[str, dict[str, tuple[str, float | None]]]:
    print("Stage D8a - selection on seed 7 (lowest validation NLL at step 1000)")
    by_key = {key(r): r for r in rows}
    chosen: dict[str, dict[str, tuple[str, float | None]]] = {}
    for schedule, suffix in SCHEDULES.items():
        chosen[schedule] = {}
        for family, configs in FAMILIES.items():
            candidates = [(k, r["val_nll"]) for k, r in by_key.items() if k[0] in {c + suffix for c in configs}]
            for k, v in sorted(candidates, key=lambda kv: kv[1]):
                print(f"  {schedule:8s} {family:15s} {arm_name(k):38s} {v:.6f}")
            best = min(candidates, key=lambda kv: kv[1])[0]
            chosen[schedule][family] = best
            print(f"  {schedule:8s} {family:15s} -> selected {arm_name(best)}")
    print()
    print("Machine check against the record (seed 7; expected agreement within 0.01):")
    for k, ref in RECORD.items():
        got = by_key[k]["val_nll"]
        flag = "ok" if abs(got - ref) <= 0.01 else "GAP > 0.01"
        print(f"  {arm_name(k):24s} record {ref:.6f}  here {got:.6f}  diff {got - ref:+.6f}  {flag}")
    print()
    return chosen


def bound_check(rows: list[dict]) -> tuple[bool, float, float]:
    ok = True
    worst_spec = 0.0
    worst_margin = math.inf
    for r in rows:
        for h in r["history"]:
            worst_spec = max(worst_spec, h["hidden_spectral_norm_max"])
            if h["hidden_spectral_norm_max"] > CAP + 1e-4:
                ok = False
            bound = h.get("att_logit_bound")
            if bound is None or math.isnan(bound):
                ok = False
                continue
            worst_margin = min(worst_margin, bound - h["att_logit_max"])
            if h["att_logit_max"] > bound:
                ok = False
    return ok, worst_spec, worst_margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path(__file__).resolve().parents[1] / "runs" / "d8")
    args = parser.parse_args()
    chosen = select(load(args.runs / "d8a_selection.jsonl"))
    rows = load(args.runs / "d8b_eval.jsonl")
    by_arm: dict[tuple[str, float | None], list[dict]] = {}
    for r in rows:
        by_arm.setdefault(key(r), []).append(r)

    print("Stage D8b - fresh seeds")
    verdicts: dict[str, str] = {}
    for schedule in SCHEDULES:
        arms = chosen[schedule]
        nll = {fam: {r["seed"]: r["val_nll"] for r in by_arm.get(k, [])} for fam, k in arms.items()}
        print(f"\n  [{schedule}]")
        for fam, k in arms.items():
            group = by_arm.get(k, [])
            vals = [r["val_nll"] for r in group]
            if not vals:
                print(f"  {fam:15s} {arm_name(k):38s} no runs")
                continue
            spec0 = statistics.fmean(r["history"][0]["hidden_spectral_norm_max"] for r in group)
            spec1 = statistics.fmean(r["hidden_spectral_norm_max_final"] for r in group)
            peak = statistics.fmean(r["att_logit_max_peak"] for r in group)
            sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            print(f"  {fam:15s} {arm_name(k):38s} seeds {sorted(r['seed'] for r in group)}  mean NLL {statistics.fmean(vals):.6f} (sd {sd:.6f})  "
                  f"spectral {spec0:.3f} -> {spec1:.3f} (x{spec1 / spec0:.2f})  peak att logit {peak:.2f}")
        if any(len(v) < 2 for v in nll.values()):
            print("  incomplete; cannot decide this schedule")
            continue

        d, lo, hi, n = paired(nll["MONA"], nll["Muon"])
        q1 = interval_verdict(lo, hi, "REPLICATED", "NOT REPLICATED", "REVERSED")
        print(f"  Q1 delta(MONA, Muon) = {d:+.6f}  95% CI [{lo:+.6f}, {hi:+.6f}]  n={n}  -> {q1}")

        means = {fam: statistics.fmean(nll[fam].values()) for fam in ("Muon", "MONA")}
        ref = min(means, key=means.get)
        dervish_rows = by_arm[arms["Dervish"]] + by_arm[arms["Dervish + MONA"]]
        ok, worst_spec, worst_margin = bound_check(dervish_rows)
        d, lo, hi, n = paired(nll["Dervish"], nll[ref])
        if not ok or d > PARTIAL:
            q2 = "KILL"
        elif d > PARITY:
            q2 = "PARTIAL"
        else:
            q2 = "PASS"
        print(f"  Q2 reference R = {ref} (mean {means[ref]:.6f}); delta(Dervish, R) = {d:+.6f}  95% CI [{lo:+.6f}, {hi:+.6f}]  "
              f"(parity line +{PARITY:.3f}, partial line +{PARTIAL:.3f})  -> {q2}")
        print(f"  Q3 bound over every evaluation of every Dervish and Dervish + MONA run: spectral max {worst_spec:.6f} (cap {CAP}), "
              f"smallest (bound - measured attention logit) {worst_margin:.4f} -> {'holds' if ok else 'FAILS'}")
        d, lo, hi, n = paired(nll["Dervish + MONA"], nll["Dervish"])
        q4 = interval_verdict(lo, hi, "COMPOSES", "NEUTRAL", "HURTS")
        print(f"  Q4 delta(Dervish + MONA, Dervish) = {d:+.6f}  95% CI [{lo:+.6f}, {hi:+.6f}]  -> {q4}")
        verdicts[schedule] = f"Q1 {q1} | Q2 {q2} | Q3 bound {'holds' if ok else 'FAILS'} | Q4 {q4}"

    print("\nVerdicts:")
    for schedule, line in verdicts.items():
        print(f"  {schedule:8s} {line}{'   (primary)' if schedule == 'decay' else ''}")
    primary = verdicts.get("decay", "")
    if "Q2 PASS" in primary:
        print("Decision: Q2 PASS on the primary schedule -> prepare for open source.")
    elif primary:
        print("Decision: Q2 not PASS on the primary schedule -> audit.")


if __name__ == "__main__":
    main()
