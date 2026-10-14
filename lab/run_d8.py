"""Stage D8 launcher: lab/compare.py jobs in parallel, one (config, seed, beta_a) per process.

Usage, from the repository root (docs/HYPOTHESIS-D002.md):

    python lab/run_d8.py . d8a runs/d8/d8a --workers 8 --threads 2
    python lab/run_d8.py . d8b runs/d8/d8b --workers 8 --threads 2 --select
        muon_lr01 mona_lr01:0.99 dervish_tuned dervish_tuned_mona:0.99
        muon_decay mona_lr01_decay:0.98 dervish_tuned_decay dervish_tuned_mona_decay:0.99

(the --select list is one command line, wrapped here).

Each job writes <out_dir>/<tag>.jsonl; finished jobs are skipped on a rerun. The
--select list for d8b is what `python lab/check_d002.py` derives from the d8a
files. Merge with `cat runs/d8/d8a/*.jsonl > runs/d8/d8a_selection.jsonl` (and
likewise d8b_eval.jsonl) before running lab/check_d002.py. The first argument is
the directory the jobs run in, so a frozen copy of the code can be used while
the working tree changes; the Stage D8 runs used such a copy.
"""

import argparse
import itertools
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

COMMON = [
    "--corpus", "lab/corpus/tiny_shakespeare_raw.txt", "--normalize",
    "--eval-interval", "100", "--eval-mode", "full",
    "--block-size", "128", "--batch-size", "32", "--n-layer", "4", "--n-head", "4", "--n-embd", "128",
    "--steps", "1000", "--device", "cpu", "--ns-dtype", "float32", "--quiet",
]
BETAS = [0.9, 0.95, 0.98, 0.99]


def d8a_jobs():
    jobs = []
    for suffix in ["", "_decay"]:
        for cfg in ["muon", "muon_lr01", "dervish_tuned"]:
            jobs.append((cfg + suffix, 7, None))
        for cfg, beta in itertools.product(["mona", "mona_lr01", "dervish_tuned_mona"], BETAS):
            jobs.append((cfg + suffix, 7, beta))
    return jobs


def d8b_jobs(selected):
    # selected: list of (config, beta or None), e.g. [("muon_lr01_decay", None), ("mona_decay", 0.95)]
    return [(cfg, seed, beta) for cfg, beta in selected for seed in [23, 29, 31, 37, 41]]


def tag(job):
    cfg, seed, beta = job
    return f"{cfg}" + (f"_b{beta}" if beta is not None else "") + f"_s{seed}"


def run(job, snapshot, out_dir, threads):
    cfg, seed, beta = job
    t = tag(job)
    cmd = [sys.executable, "lab/compare.py", *COMMON, "--threads", str(threads), "--configs", cfg, "--seeds", str(seed),
           "--out", str(out_dir / f"{t}.jsonl"), "--summary", str(out_dir / "md" / f"{t}.md"), "--title", t]
    if beta is not None:
        cmd += ["--sweep-key", "mona_beta", "--sweep-values", str(beta)]
    start = time.time()
    proc = subprocess.run(cmd, cwd=snapshot, capture_output=True, text=True)
    (out_dir / "logs" / f"{t}.log").write_text(" ".join(cmd) + "\n\n" + proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    return t, proc.returncode, time.time() - start, proc.stdout.strip().splitlines()[-2:] if proc.stdout else proc.stderr[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("stage")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--select", nargs="*", default=[], help="d8b: config[:beta] entries")
    args = ap.parse_args()
    (args.out_dir / "md").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(parents=True, exist_ok=True)
    if args.stage == "d8a":
        jobs = d8a_jobs()
    elif args.stage == "d8b":
        selected = [(s.split(":")[0], float(s.split(":")[1]) if ":" in s else None) for s in args.select]
        jobs = d8b_jobs(selected)
    else:
        raise SystemExit("unknown stage")
    jobs = [j for j in jobs if not (args.out_dir / f"{tag(j)}.jsonl").exists()]
    print(f"{len(jobs)} jobs, {args.workers} workers x {args.threads} threads", flush=True)
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, j, args.snapshot, args.out_dir, args.threads) for j in jobs]
        for done, fut in enumerate(as_completed(futures), 1):
            t, rc, secs, tail = fut.result()
            print(f"[{done}/{len(jobs)} {time.time() - start:7.0f}s] {t} rc={rc} {secs:5.0f}s {tail}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
