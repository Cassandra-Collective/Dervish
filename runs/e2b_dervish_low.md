# E2b Dervish LR Sweep, lower grid (post-hoc)

Command:

```
python lab/compare.py --corpus lab/corpus/tiny_shakespeare_raw.txt --normalize --eval-interval 100 --eval-mode full --block-size 128 --batch-size 32 --n-layer 4 --n-head 4 --n-embd 128 --threads 2 --quiet --steps 600 --seeds 7 --configs dervish --sweep-key dervish_lr --sweep-values 0.003125 0.00625 --out runs/e2b_dervish_low.jsonl --summary runs/e2b_dervish_low.md --title 'E2b Dervish LR Sweep, lower grid (post-hoc)'
```

| Config | Sweep | Seeds | Trainable params | Mean val NLL | Min val NLL | Max val NLL | Mean bits/char | Peak att logit (mean) | Final spectral max (mean) | Final stable rank (mean) | Final out logit max (mean) | Diverged | Mean train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dervish | 0.0031 | 1 | 802596 | 1.747590 | 1.747590 | 1.747590 | 2.521240 | 8.82 | 0.655 | 73.01 | 10.73 | 0 | 180.7 |
| dervish | 0.0063 | 1 | 802596 | 1.698934 | 1.698934 | 1.698934 | 2.451044 | 9.09 | 0.631 | 74.14 | 10.73 | 0 | 180.4 |

Raw rows:

| Config | Seed | Sweep | val NLL | bits/char | peak att logit | final att logit | final spectral max | final stable rank | final out logit max | diverged at | train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dervish | 7 | 0.0031 | 1.747590 | 2.521240 | 8.82 | 8.82 | 0.655 | 73.01 | 10.73 | · | 180.7 |
| dervish | 7 | 0.0063 | 1.698934 | 2.451044 | 9.09 | 9.09 | 0.631 | 74.14 | 10.73 | · | 180.4 |

Evaluation curves (means over seeds):

`dervish @ 0.0031`

| step | val NLL | att logit max | out logit max | spectral max | stable rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.5793 | 0.29 | 0.92 | 0.504 | 84.58 |
| 100 | 2.4277 | 0.55 | 6.63 | 0.517 | 79.26 |
| 200 | 2.2890 | 1.92 | 7.53 | 0.527 | 75.08 |
| 300 | 2.0614 | 4.73 | 8.11 | 0.533 | 74.12 |
| 400 | 1.8917 | 7.15 | 9.19 | 0.587 | 73.64 |
| 500 | 1.8083 | 8.28 | 9.73 | 0.624 | 73.07 |
| 600 | 1.7476 | 8.82 | 10.73 | 0.655 | 73.01 |

`dervish @ 0.0063`

| step | val NLL | att logit max | out logit max | spectral max | stable rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.5793 | 0.29 | 0.92 | 0.504 | 84.58 |
| 100 | 2.3963 | 0.94 | 6.79 | 0.508 | 79.89 |
| 200 | 2.1702 | 3.65 | 7.60 | 0.521 | 75.81 |
| 300 | 1.9340 | 6.07 | 8.59 | 0.536 | 75.00 |
| 400 | 1.8076 | 7.38 | 9.81 | 0.556 | 74.42 |
| 500 | 1.7418 | 8.64 | 9.85 | 0.589 | 74.05 |
| 600 | 1.6989 | 9.09 | 10.73 | 0.631 | 74.14 |


Matrix wall-clock: 413.9 s · platform: Linux-6.18.44-fc-v80-x86_64-with-glibc2.39 · torch 2.14.1+cu130 · device cpu · threads 2
