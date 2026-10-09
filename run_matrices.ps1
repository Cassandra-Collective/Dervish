# Reproduce every matrix behind docs/RESULTS.md. Windows PowerShell 7. See run_matrices.sh for Linux/macOS.
# Usage: .\run_matrices.ps1 [e1|e2|e3|all]   (CUDA: add -Device cuda; the Newton-Schulz dtype then defaults to bfloat16)
param([string]$Stage = "all", [string]$Device = "cpu", [int]$Threads = 4)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$text = @("--corpus", "lab/corpus/tiny_shakespeare_raw.txt", "--normalize", "--eval-interval", "100", "--eval-mode", "full", "--block-size", "128", "--batch-size", "32", "--n-layer", "4", "--n-head", "4", "--n-embd", "128", "--threads", "$Threads", "--device", $Device, "--quiet")
$copy = @("--corpus", "lab/corpus/long_context_seed.txt", "--eval-interval", "100", "--eval-mode", "full", "--block-size", "96", "--batch-size", "32", "--n-layer", "4", "--n-head", "4", "--n-embd", "128", "--threads", "$Threads", "--device", $Device, "--quiet")
if ($Stage -in @("e1", "all")) {
  python lab/compare.py @text --steps 1000 --seeds 7 11 19 --configs adamw muon muon_wd dervish dervish_nocap --out runs/e1_text.jsonl --summary runs/e1_text.md --title "E1 Text Summary"
}
if ($Stage -in @("e2", "all")) {
  python lab/compare.py @text --steps 600 --seeds 7 --configs muon --sweep-key muon_lr --sweep-values 0.005 0.01 0.02 0.04 0.08 0.16 --out runs/e2_muon.jsonl --summary runs/e2_muon.md --title "E2 Muon LR Sweep"
  python lab/compare.py @text --steps 600 --seeds 7 --configs dervish --sweep-key dervish_lr --sweep-values 0.0125 0.025 0.05 0.1 0.2 0.4 --out runs/e2_dervish.jsonl --summary runs/e2_dervish.md --title "E2 Dervish LR Sweep"
  python lab/compare.py @text --steps 600 --seeds 7 --configs adamw --sweep-key lr --sweep-values 0.00025 0.0005 0.001 0.002 0.004 0.008 --out runs/e2_adamw.jsonl --summary runs/e2_adamw.md --title "E2 AdamW LR Sweep"
}
if ($Stage -in @("e3", "all")) {
  python lab/compare.py @copy --steps 1000 --seeds 7 11 19 --copy-probe-marker "answer=" --configs adamw muon muon_wd dervish adamw_copyw muon_copyw muon_wd_copyw dervish_copyw --out runs/e3_copy.jsonl --summary runs/e3_copy.md --title "E3 Copy Behaviour Summary"
}
Write-Host "done: $Stage"
