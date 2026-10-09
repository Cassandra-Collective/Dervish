#!/usr/bin/env bash
# One-command check that the package works on this machine: unit tests, both examples,
# and a 20-step training smoke test per optimizer. About two minutes on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q tests
python examples/nearest_unitary.py | tail -4
python examples/experts_demo.py | tail -5
for opt in adamw muon dervish; do
  python lab/train.py --corpus lab/corpus/tiny_shakespeare_raw.txt --normalize --optimizer "$opt" --steps 20 --eval-interval 10 --log-every 0 --label "smoke_$opt" --quiet \
    | python -c "import json,sys; r=json.load(sys.stdin); print(f\"smoke {r['label']}: val_nll {r['val_nll']:.4f} spectral_max {r['hidden_spectral_norm_max_final']:.3f} train_s {r['train_seconds']:.1f}\")"
done
python lab/check_claims.py | tail -1
echo "verify: ok"
