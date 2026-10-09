# One-command check that the package works on this machine (Windows PowerShell 7).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m pytest -q tests
python examples/nearest_unitary.py | Select-Object -Last 4
python examples/experts_demo.py | Select-Object -Last 5
foreach ($opt in @("adamw", "muon", "dervish")) {
  python lab/train.py --corpus lab/corpus/tiny_shakespeare_raw.txt --normalize --optimizer $opt --steps 20 --eval-interval 10 --log-every 0 --label "smoke_$opt" --quiet |
    python -c "import json,sys; r=json.load(sys.stdin); print(f\"smoke {r['label']}: val_nll {r['val_nll']:.4f} spectral_max {r['hidden_spectral_norm_max_final']:.3f} train_s {r['train_seconds']:.1f}\")"
}
python lab/check_claims.py | Select-Object -Last 1
Write-Host "verify: ok"
