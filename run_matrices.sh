#!/usr/bin/env bash
# Reproduce every matrix behind docs/RESULTS.md. Linux/macOS. See run_matrices.ps1 for Windows.
# Each run writes runs/<stage>.jsonl (one report per line) and runs/<stage>.md (summary).
set -euo pipefail
cd "$(dirname "$0")"
THREADS="${THREADS:-4}"
COMMON_TEXT=(--corpus lab/corpus/tiny_shakespeare_raw.txt --normalize --eval-interval 100 --eval-mode full --block-size 128 --batch-size 32 --n-layer 4 --n-head 4 --n-embd 128 --threads "$THREADS" --quiet)
COMMON_COPY=(--corpus lab/corpus/long_context_seed.txt --eval-interval 100 --eval-mode full --block-size 96 --batch-size 32 --n-layer 4 --n-head 4 --n-embd 128 --threads "$THREADS" --quiet)

stage="${1:-all}"

if [[ "$stage" == "e1" || "$stage" == "all" ]]; then
  python lab/compare.py "${COMMON_TEXT[@]}" --steps 1000 --seeds 7 11 19 \
    --configs adamw muon muon_wd dervish dervish_nocap \
    --out runs/e1_text.jsonl --summary runs/e1_text.md --title "E1 Text Summary"
fi
if [[ "$stage" == "e2" || "$stage" == "all" ]]; then
  python lab/compare.py "${COMMON_TEXT[@]}" --steps 600 --seeds 7 --configs muon --sweep-key muon_lr --sweep-values 0.005 0.01 0.02 0.04 0.08 0.16 \
    --out runs/e2_muon.jsonl --summary runs/e2_muon.md --title "E2 Muon LR Sweep"
  python lab/compare.py "${COMMON_TEXT[@]}" --steps 600 --seeds 7 --configs dervish --sweep-key dervish_lr --sweep-values 0.0125 0.025 0.05 0.1 0.2 0.4 \
    --out runs/e2_dervish.jsonl --summary runs/e2_dervish.md --title "E2 Dervish LR Sweep"
  python lab/compare.py "${COMMON_TEXT[@]}" --steps 600 --seeds 7 --configs adamw --sweep-key lr --sweep-values 0.00025 0.0005 0.001 0.002 0.004 0.008 \
    --out runs/e2_adamw.jsonl --summary runs/e2_adamw.md --title "E2 AdamW LR Sweep"
fi
if [[ "$stage" == "e3" || "$stage" == "all" ]]; then
  python lab/compare.py "${COMMON_COPY[@]}" --steps 1000 --seeds 7 11 19 --copy-probe-marker "answer=" \
    --configs adamw muon muon_wd dervish adamw_copyw muon_copyw muon_wd_copyw dervish_copyw \
    --out runs/e3_copy.jsonl --summary runs/e3_copy.md --title "E3 Copy Behaviour Summary"
fi
[[ "$stage" == "e4" || "$stage" == "e5" || "$stage" == "e2b" ]] || echo "done: $stage"

# E4 (post-hoc, launched after E1 was read): the spectrum learning-rate sweep the hypothesis
# named as the follow-up to a KILL on claim 2. Seed 7, same model and budget as E1.
if [[ "$stage" == "e4" ]]; then
  python lab/compare.py "${COMMON_TEXT[@]}" --steps 1000 --seeds 7 --configs dervish dervish_nocap --sweep-key dervish_scale_lr --sweep-values 0.003 0.01 0.03 0.1 \
    --out runs/e4_scale_lr.jsonl --summary runs/e4_scale_lr.md --title "E4 Dervish Spectrum LR Sweep (post-hoc)"
  echo "done: e4"
fi

# E5 (post-hoc, after E2 and E4 were read): smaller rotation angles, a faster spectrum, and an
# explicit linear learning-rate decay (Muon gets the same decay as a fair control). Seed 7, 1000 steps.
if [[ "$stage" == "e5" ]]; then
  rm -f runs/e5_tuned.jsonl
  T=(python lab/train.py "${COMMON_TEXT[@]}" --steps 1000 --seed 7 --out runs/e5_tuned.jsonl)
  "${T[@]}" --optimizer dervish --dervish-lr 0.0125 --dervish-scale-lr 0.01 --dervish-cap 1.0 --label dervish_a0125_s01_cap
  "${T[@]}" --optimizer dervish --dervish-lr 0.0125 --dervish-scale-lr 0.01 --dervish-cap 0   --label dervish_a0125_s01_nocap
  "${T[@]}" --optimizer dervish --dervish-lr 0.00625 --dervish-scale-lr 0.01 --dervish-cap 1.0 --label dervish_a00625_s01_cap
  "${T[@]}" --optimizer dervish --dervish-lr 0.00625 --dervish-scale-lr 0.01 --dervish-cap 0   --label dervish_a00625_s01_nocap
  "${T[@]}" --optimizer dervish --dervish-lr 0.0125 --dervish-scale-lr 0.01 --dervish-cap 1.0 --lr-decay-to 0.1 --label dervish_a0125_s01_cap_decay
  "${T[@]}" --optimizer muon --muon-lr 0.02 --lr-decay-to 0.1 --label muon_lr02_decay
  "${T[@]}" --optimizer muon --muon-lr 0.01 --label muon_lr01
  python lab/summarize_jsonl.py --jsonl runs/e5_tuned.jsonl --summary runs/e5_tuned.md --title "E5 Dervish Tuned (post-hoc)" --command "./run_matrices.sh e5"
  echo "done: e5"
fi

# E2b (post-hoc): extend the Dervish rotation-angle grid below E2's smallest point. Seed 7, 600 steps.
if [[ "$stage" == "e2b" ]]; then
  python lab/compare.py "${COMMON_TEXT[@]}" --steps 600 --seeds 7 --configs dervish --sweep-key dervish_lr --sweep-values 0.003125 0.00625 \
    --out runs/e2b_dervish_low.jsonl --summary runs/e2b_dervish_low.md --title "E2b Dervish LR Sweep, lower grid (post-hoc)"
  echo "done: e2b"
fi
