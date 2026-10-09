"""Write the compare.py-style markdown summary for any runs/*.jsonl (for stages
assembled from individual lab/train.py invocations)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.compare import summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--command", default="see run_matrices.sh")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.summary.write_text(summarize(rows, args.title, args.command), encoding="utf-8")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
