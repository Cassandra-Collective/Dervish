# Results · Dervish lab, staged record

Every number here traces to a file under `runs/`. Platform for every stage:
a Linux x86-64 cloud container (kernel `6.18`, glibc `2.39`), `4` CPU threads of an Intel Xeon at `2.10 GHz`, `15 GB` RAM, Python `3.13.16`, PyTorch `2.14.1` (a CUDA build with no GPU present, so every run is CPU float32 and every Newton-Schulz iteration runs in float32), matplotlib `3.11.2`. Two or three matrices usually shared the four cores, so the wall-clock columns are upper bounds for a quiet machine. Seeds `7 11 19` unless stated. Validation NLL is the
chunked deterministic evaluation over the whole validation split
(`--eval-mode full`), in nats per character; bits/char is NLL divided by
`ln 2`. Verdicts are computed by `lab/check_claims.py` against the lines fixed
in `docs/HYPOTHESIS-D001.md` before any run.

STAGES_PLACEHOLDER
