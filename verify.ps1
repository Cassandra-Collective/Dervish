# One-command check that the package works on this machine (Windows PowerShell 7).
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true  # PowerShell 7.3+: a failing python stops the script
Set-Location $PSScriptRoot
python -m pytest -q tests
python examples/nearest_unitary.py | Select-Object -Last 4
python examples/experts_demo.py | Select-Object -Last 5
foreach ($opt in @("adamw", "muon", "dervish")) {
  python lab/train.py --corpus lab/corpus/tiny_shakespeare_raw.txt --normalize --optimizer $opt --steps 20 --eval-interval 10 --log-every 0 --label "smoke_$opt" --quiet |
    python -c "import json,sys; r=json.load(sys.stdin); print('smoke %s: val_nll %.4f spectral_max %.3f train_s %.1f' % (r['label'], r['val_nll'], r['hidden_spectral_norm_max_final'], r['train_seconds']))"
}
python lab/check_claims.py | Select-Object -Last 1
Write-Host "verify: ok"
